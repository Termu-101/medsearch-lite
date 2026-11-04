"""
Phase 1 - Data collection.

Fetches PubMed abstracts on diabetes via the Biopython Entrez API,
extracts metadata (PMID, title, abstract text, authors, journal,
publication date, publication type, MeSH terms), tags each abstract
with an evidence level derived from its publication type, and saves
the results as JSON to data/raw/.
"""

# Import and Setup

import os
import json
import time
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()
Entrez.email = "terrmu10.10@gmail.com"
Entrez.api_key = os.getenv("NCBI_API_KEY")

# Search PubMed for matching PMIDs

def search_pubmed_ids(query, max_results=500):
    """Search PubMed and return a list of PMIDs matching the query, sorted by relevance."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


# Fetch full records in batches

def fetch_records_in_batches(pmid_list, batch_size=100):
    """Fetch full PubMed records for a list of PMIDs, in batches, returning parsed XML records."""
    all_records = []
    for start in range(0, len(pmid_list), batch_size):
        batch = pmid_list[start:start + batch_size]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="xml")
        data = Entrez.read(handle)
        handle.close()
        all_records.extend(data["PubmedArticle"])
        print(f"Fetched {start + len(batch)}/{len(pmid_list)} records...")
        time.sleep(0.34)  # stay under NCBI rate limit
    return all_records

# Map publication type → evidence level

def get_evidence_level(pub_types):
    """Given a list of publication type strings, return an evidence level 1-5."""
    pub_types_lower = [p.lower() for p in pub_types]
    if any("meta-analysis" in p or "systematic review" in p for p in pub_types_lower):
        return 1
    if any("randomized controlled trial" in p for p in pub_types_lower):
        return 2
    if any("clinical trial" in p or "multicenter study" in p for p in pub_types_lower):
        return 3
    if any("observational study" in p or "cohort" in p for p in pub_types_lower):
        return 4
    return 5  # case reports, reviews, everything else


# Parse one record into a clean dict

def parse_record(article):
    """Extract PMID, title, abstract, authors, journal, date, pub types, MeSH terms from one PubmedArticle XML record."""
    medline = article["MedlineCitation"]
    article_data = medline["Article"]

    pmid = str(medline["PMID"])
    title = article_data.get("ArticleTitle", "")

    abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
    abstract_text = " ".join(str(part) for part in abstract_parts)

    authors = []
    for author in article_data.get("AuthorList", []):
        if "LastName" in author and "ForeName" in author:
            authors.append(f"{author['ForeName']} {author['LastName']}")

    journal = article_data.get("Journal", {}).get("Title", "")

    pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = pub_date.get("Year", "")

    pub_types = [str(pt) for pt in article_data.get("PublicationTypeList", [])]

    mesh_terms = [str(mesh["DescriptorName"]) for mesh in medline.get("MeshHeadingList", [])]

    return {
        "pmid": pmid,
        "title": str(title),
        "abstract": abstract_text,
        "authors": authors,
        "journal": str(journal),
        "publication_year": year,
        "publication_types": pub_types,
        "mesh_terms": mesh_terms,
        "evidence_level": get_evidence_level(pub_types),
    }

# Filter out abstract-less records

def filter_valid_records(records):
    """Keep only records that actually have non-empty abstract text."""
    return [r for r in records if r["abstract"].strip()]


# Save to disk

def save_to_json(records, filepath):
    """Save a list of record dicts as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records to {filepath}")


# Tie it together in main()

def main():
    query = "diabetes mellitus[MeSH Terms]"
    pmids = search_pubmed_ids(query, max_results=500)
    print(f"Found {len(pmids)} PMIDs")

    raw_records = fetch_records_in_batches(pmids)
    parsed = [parse_record(article) for article in raw_records]
    valid = filter_valid_records(parsed)

    save_to_json(valid, "data/raw/pubmed_diabetes_500.json")

if __name__ == "__main__":
    main()

