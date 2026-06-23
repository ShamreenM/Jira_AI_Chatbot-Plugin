import re
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.document import Document
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder


# =========================================
# DATE INDEX CACHE  (NEW)
# =========================================
# The date-query branch needs to scan the full collection (Created /
# Updated / comment text). That scan is the slow part — not the matching
# itself. Caching it in memory means it only runs ONCE per app session;
# every date question after the first reuses it instantly.
#
# NOTE: this cache lives only for the lifetime of the running app
# process. If your underlying Chroma DB changes (new tickets ingested)
# while the app keeps running, the cache won't see those changes until
# the app is restarted.

_date_index_cache = None


def _get_full_collection(vectorstore, batch_size=1500):

    global _date_index_cache

    if _date_index_cache is not None:
        return _date_index_cache

    offset = 0
    page = 0

    all_metadatas = []
    all_documents = []

    while True:

        batch = vectorstore.get(
            include=["metadatas", "documents"],
            limit=batch_size,
            offset=offset
        )

        batch_metadatas = batch.get("metadatas") or []
        batch_documents = batch.get("documents") or []

        if not batch_metadatas:
            break

        all_metadatas.extend(batch_metadatas)
        all_documents.extend(batch_documents)

        page += 1
        print(f"📦 Scanning collection for date index... {len(all_metadatas)} records loaded so far (page {page})")

        if len(batch_metadatas) < batch_size:
            break

        offset += batch_size

    print(f"✅ Date index built: {len(all_metadatas)} total records")

    _date_index_cache = (all_metadatas, all_documents)

    return _date_index_cache


# =========================================
# QUERY GENERATION
# =========================================

def generate_queries(query, llm):

    prompt = f"""
You are a JIRA expert.

Rewrite the query into 4 search queries.

Focus on:
- unresolved tickets
- sprint carry forward issues
- incomplete stories
- pending tasks
- backlog spillover
- issue investigation

Query:
{query}
"""

    response = llm.invoke(prompt)

    return [
        q.strip("- ").strip()
        for q in response.content.split("\n")
        if q.strip()
    ]


# =========================================
# HYBRID SEARCH
# =========================================

def hybrid_search_with_scores(query, vectorstore, k=40):

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    docs_from_vectorstore = [
        Document(
            page_content=doc.page_content,
            metadata=doc.metadata
        )
        for doc in vectorstore.similarity_search("", k=100)
    ]

    bm25_retriever = BM25Retriever.from_documents(
        docs_from_vectorstore
    )

    bm25_retriever.k = 20

    dense_results = dense_retriever.invoke(query)

    sparse_results = bm25_retriever.invoke(query)

    scores = {}

    doc_map = {}

    # Dense scores
    for rank, doc in enumerate(dense_results):

        key = doc.page_content

        score = 0.7 / (rank + 1)

        scores[key] = scores.get(key, 0) + score

        doc_map[key] = doc

    # Sparse scores
    for rank, doc in enumerate(sparse_results):

        key = doc.page_content

        score = 0.3 / (rank + 1)

        scores[key] = scores.get(key, 0) + score

        doc_map[key] = doc

    sorted_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (doc_map[key], score)
        for key, score in sorted_docs[:k]
    ]


# =========================================
# DYNAMIC TOP K
# =========================================

def dynamic_top_k(results, query, max_k=10):

    filtered_docs = []

    if any(
        word in query.lower()
        for word in ["why", "reason", "analysis"]
    ):
        threshold = 0.05
    else:
        threshold = 0.2

    for doc, score in results:

        if score >= threshold:
            filtered_docs.append(doc)

        if len(filtered_docs) >= max_k:
            break

    return filtered_docs


# =========================================
# RERANKING
# =========================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_documents(query, docs, top_k=5):

    if len(docs) == 0:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc for doc, _ in scored_docs[:top_k]
    ]


# =========================================
# STRUCTURED METADATA FILTER  (NEW)
# =========================================
# Ported over unchanged from the earlier fix. Detects "list/show/give all
# X" style questions that name an exact status / priority / resolution /
# severity value, and builds a direct Chroma metadata filter for them
# instead of relying on fuzzy semantic + BM25 matching, which often
# returns nothing for these because the field values rarely appear
# verbatim inside the issue description text.
#
# NOTE: field names below ('status', 'priority', 'Resolution',
# 'Custom field (Symptom Severity)') match the keys already used in your
# original context-builder. This version's context-builder doesn't print
# Resolution or Symptom Severity, but the metadata filter itself doesn't
# depend on that — it filters directly against whatever is stored in
# Chroma, so it still works as long as those keys exist there.

def detect_structured_filter(query):

    q = query.lower()

    STATUS_VALUES = [
        "Needs Triage", "Closed", "Gathering Interest", "Gathering Impact",
        "Long Term Backlog", "Short Term Backlog", "In Progress",
        "Under Consideration", "Future Consideration"
    ]
    PRIORITY_VALUES = ["Low", "Medium", "High", "Highest"]
    RESOLUTION_VALUES = [
        "Fixed", "Duplicate", "Cannot Reproduce", "Spam", "Not a bug",
        "Invalid", "Answered", "Incorrectly Filed", "Resolved Locally"
    ]
    SEVERITY_VALUES = [
        "Severity 1 - Critical", "Severity 2 - Major", "Severity 3 - Minor",
        "Critical", "Major", "Minor"
    ]

    def find_matches(values):
        matches = []
        for v in values:
            # \b...\b avoids "High" wrongly matching inside "Highest"
            pattern = r"\b" + re.escape(v.lower()) + r"\b"
            if re.search(pattern, q):
                matches.append(v)
        return matches

    status_matches = find_matches(STATUS_VALUES)
    priority_matches = find_matches(PRIORITY_VALUES)
    resolution_matches = find_matches(RESOLUTION_VALUES)
    severity_matches = find_matches(SEVERITY_VALUES)

    conditions = []
    description_parts = []
    breakdown = []  # list of (label, single-value where-filter) for exact counts

    if status_matches:
        conditions.append({"status": {"$in": status_matches}})
        description_parts.append(f"status in {status_matches}")
        for v in status_matches:
            breakdown.append((f"status = {v}", {"status": v}))

    if priority_matches:
        conditions.append({"priority": {"$in": priority_matches}})
        description_parts.append(f"priority in {priority_matches}")
        for v in priority_matches:
            breakdown.append((f"priority = {v}", {"priority": v}))

    if resolution_matches:
        conditions.append({"Resolution": {"$in": resolution_matches}})
        description_parts.append(f"resolution in {resolution_matches}")
        for v in resolution_matches:
            breakdown.append((f"resolution = {v}", {"Resolution": v}))

    if severity_matches:
        conditions.append({"Custom field (Symptom Severity)": {"$in": severity_matches}})
        description_parts.append(f"severity in {severity_matches}")
        for v in severity_matches:
            breakdown.append((f"severity = {v}", {"Custom field (Symptom Severity)": v}))

    # "still open" / "unresolved" with no explicit status named -> exclude Closed
    if not status_matches and any(w in q for w in ["still open", "not closed", "unresolved", "pending"]):
        conditions.append({"status": {"$ne": "Closed"}})
        description_parts.append("status != Closed")

    if not conditions:
        return None, None, []

    filter_dict = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    return filter_dict, ", ".join(description_parts), breakdown


# =========================================
# RETRIEVE DOCS
# =========================================

def retrieve_docs(query, apiKey):

    # =====================================
    # EMBEDDINGS
    # =====================================

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    # =====================================
    # LOAD VECTORSTORE
    # =====================================

    vectorstore = Chroma(
        persist_directory="./chroma_db_backup",
        embedding_function=embedding
    )

    # =====================================
    # LLM
    # =====================================
    # CHANGED: was hardcoded google_api_key="xxx" — now uses the apiKey
    # parameter already being passed in from app.py (loaded there from
    # key.env), exactly the same pattern your OpenAI key already used.

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=apiKey,
        temperature=0.3
    )

    # Holds exact-count stats from the structured filter path (if used),
    # prepended to the context below. Stays empty for every other path.
    extra_context_prefix = ""

    # =====================================
    # DATE QUERY
    # =====================================

    date_match = re.search(
        r"\d{1,2}/[A-Za-z]{3}/\d{4}",
        query
    )

    if date_match:

        target_date = date_match.group()

        # CHANGED: now pulls from the cached scan (see _get_full_collection
        # above) instead of re-fetching the full collection on every date
        # question. First date question in a session still pays the full
        # scan cost; every one after that is near-instant.

        all_metadatas, all_documents = _get_full_collection(vectorstore)

        matched_issue_keys = set()

        for meta, content in zip(all_metadatas, all_documents):

            created = meta.get("Created") or ""
            updated = meta.get("Updated") or ""
            comments = content or ""

            if (
                target_date.lower() in created.lower()
                or target_date.lower() in updated.lower()
                or target_date.lower() in comments.lower()
            ):
                matched_issue_keys.add(meta.get("issue_key"))

        if len(matched_issue_keys) == 0:

            return (
                f"No issues found with activity "
                f"on {target_date}"
            )

        return "\n".join(sorted(matched_issue_keys))

    # =====================================
    # ISSUE KEY QUERY
    # =====================================

    match = re.search(
        r"[A-Z]+-\d+",
        query
    )

    if match:

        issue_key = match.group()

        docs = vectorstore.similarity_search(
            query="",
            filter={
                "issue_key": issue_key
            },
            k=5
        )

        if len(docs) == 0:

            return (
                f"Issue {issue_key} "
                f"not found in dataset."
            )

    else:

        # =================================
        # STRUCTURED METADATA FILTER  (NEW)
        # =================================
        # Only runs for extraction-style "list/show/give all X" questions
        # — NEVER for analytical (why/reason/analysis) questions. If no
        # filter is detected, execution falls straight into the original
        # QUERY EXPANSION block below, completely unchanged.

        is_analytical_query = any(
            word in query.lower()
            for word in ["why", "reason", "analysis"]
        )

        filter_dict, matched_desc, breakdown = (
            (None, None, []) if is_analytical_query
            else detect_structured_filter(query)
        )

        if filter_dict:

            # Exact counts per individually detected value (e.g. for
            # "how many" / "compare X vs Y" questions) — computed as a
            # cheap filtered count, not by counting the 25-doc sample,
            # so the LLM gets real numbers instead of guessing from a
            # partial sample.
            count_lines = []

            for label, single_filter in breakdown:
                try:
                    count_result = vectorstore.get(
                        where=single_filter,
                        include=[]
                    )
                    count_lines.append(
                        f"- {label}: {len(count_result.get('ids', []))} tickets"
                    )
                except Exception:
                    pass

            extra_context_prefix = ""

            if count_lines:
                extra_context_prefix = (
                    "EXACT COUNTS (from the full dataset, not just the "
                    "sample below):\n" + "\n".join(count_lines) + "\n\n"
                )

            filtered_results = vectorstore.similarity_search(
                query="",
                filter=filter_dict,
                k=200
            )

            total_found = len(filtered_results)

            if total_found == 0:
                return f"No tickets found matching: {matched_desc}"

            def _parse_updated(doc):
                try:
                    return datetime.strptime(doc.metadata.get("Updated", ""), "%d/%b/%Y %I:%M %p")
                except (ValueError, TypeError):
                    return datetime.min

            filtered_results.sort(key=_parse_updated, reverse=True)

            docs = filtered_results[:25]

        else:

            # =================================
            # QUERY EXPANSION  (ORIGINAL — UNCHANGED)
            # =================================

            queries = generate_queries(
                query,
                llm
            )[:3]

            all_docs = []

            for q in queries:

                results = hybrid_search_with_scores(
                    q,
                    vectorstore,
                    k=40
                )

                filtered_docs = dynamic_top_k(
                    results,
                    query,
                    max_k=10
                )

                all_docs.extend(filtered_docs)

            # =================================
            # FALLBACK
            # =================================

            if len(all_docs) == 0:

                fallback = hybrid_search_with_scores(
                    query,
                    vectorstore,
                    k=5
                )

                all_docs = [
                    doc for doc, _ in fallback
                ]

            # =================================
            # DEDUP
            # =================================

            clean_docs = [

                doc if not isinstance(doc, tuple)
                else doc[0]

                for doc in all_docs
            ]

            unique_docs = list({

                doc.page_content: doc

                for doc in clean_docs

            }.values())

            # =================================
            # RERANK
            # =================================

            docs = rerank_documents(
                query,
                unique_docs,
                top_k=5
            )

            # =================================
            # FINAL FALLBACK
            # =================================

            if len(docs) == 0:

                docs = vectorstore.similarity_search(
                    query,
                    k=5
                )

    # =====================================
    # CONTEXT BUILD
    # =====================================

    context = extra_context_prefix

    for doc in docs:

        context += f"""
Issue Key: {doc.metadata.get('issue_key')}

Issue Type: {doc.metadata.get('issue_type')}

Status: {doc.metadata.get('status')}

Project Name: {doc.metadata.get('project')}

Priority: {doc.metadata.get('priority')}

Resolution: {doc.metadata.get('Resolution')}

Symptom Severity: {doc.metadata.get('Custom field (Symptom Severity)')}

Created: {doc.metadata.get('Created')}

Updated: {doc.metadata.get('Updated')}

Description:
{doc.page_content}

"""


    # =====================================
    # INTENT
    # =====================================

    if any(
        word in query.lower()
        for word in ["why", "reason", "analysis"]
    ):
        mode = "analysis"
    else:
        mode = "extraction"

    # =====================================
    # PROMPT
    # =====================================

    if mode == "analysis":

        prompt = f"""
You are a JIRA analyst.

Analyze the issues using ONLY the fields present in the context below
(Issue Key, Issue Type, Status, Project Name, Priority, Resolution,
Symptom Severity, Created, Updated, Description). You may infer patterns
and likely causes from these fields.

Do NOT introduce or assume any concept that is not one of those fields —
for example, do not discuss sprints, sprint assignment, story points, or
velocity unless that information actually appears in the context. If the
question asks about something not tracked in this data, say plainly that
this data doesn't track it, instead of reinterpreting other fields to
imply an answer.

CONTEXT:
{context}

QUESTION:
{query}

Give a concise analysis grounded only in the fields listed above. If the
question can't be answered from those fields, say so clearly instead of
inferring beyond them.
"""

    else:

        prompt = f"""
You are a JIRA assistant.

Answer ONLY from context.

CONTEXT:
{context}

QUESTION:
{query}

If answer not found say:
No issue found.
"""

    response = llm.invoke(prompt)

    return response.content
