"""
Phase 5, Step B - Medical synonym expansion.

Defines a lookup table of diabetes-related medical synonym pairs used
to expand query terms before BM25 keyword search. The original query
is kept unmodified for semantic (embedding-based) search; only the
BM25 side benefits from expansion.
"""

MEDICAL_SYNONYMS = {
    "type 2 diabetes": ["t2dm", "type 2 diabetes mellitus", "niddm", "adult-onset diabetes"],
    "type 1 diabetes": ["t1dm", "type 1 diabetes mellitus", "iddm", "juvenile diabetes"],
    "diabetes mellitus": ["diabetes", "dm"],
    "hba1c": ["glycated hemoglobin", "glycosylated hemoglobin", "a1c"],
    "blood glucose": ["blood sugar", "glycemia", "plasma glucose"],
    "hyperglycemia": ["high blood sugar", "elevated glucose"],
    "hypoglycemia": ["low blood sugar", "low glucose"],
    "insulin resistance": ["ir"],
    "metformin": ["glucophage"],
    "sulfonylurea": ["su"],
    "glp-1 receptor agonist": ["glp-1 ra", "glucagon-like peptide-1 agonist"],
    "sglt2 inhibitor": ["sodium-glucose cotransporter-2 inhibitor", "sglt2i"],
    "dpp-4 inhibitor": ["dipeptidyl peptidase-4 inhibitor", "gliptin"],
    "insulin therapy": ["insulin treatment"],
    "diabetic nephropathy": ["diabetic kidney disease", "dkd"],
    "diabetic retinopathy": ["dr"],
    "diabetic neuropathy": ["peripheral neuropathy in diabetes"],
    "diabetic foot ulcer": ["dfu"],
    "cardiovascular disease": ["cvd", "heart disease"],
    "chronic kidney disease": ["ckd"],
    "obesity": ["overweight", "excess body weight"],
    "body mass index": ["bmi"],
    "glycemic control": ["blood sugar control", "glucose control"],
    "fasting plasma glucose": ["fpg", "fasting blood glucose", "fbg"],
    "oral glucose tolerance test": ["ogtt"],
    "continuous glucose monitoring": ["cgm"],
    "self-monitoring of blood glucose": ["smbg"],
    "randomized controlled trial": ["rct"],
    "systematic review": ["meta-analysis"],
    "gestational diabetes": ["gdm", "diabetes in pregnancy"],
    "prediabetes": ["impaired glucose tolerance", "igt", "impaired fasting glucose", "ifg"],
    "beta cell": ["islet beta cell", "pancreatic beta cell"],
    "weight loss": ["weight reduction"],
    "lifestyle intervention": ["diet and exercise intervention"],
    "bariatric surgery": ["metabolic surgery", "weight loss surgery"],
    "adverse events": ["side effects", "adverse effects"],
    "mortality": ["death rate", "all-cause mortality"],
    "hypertension": ["high blood pressure"],
    "dyslipidemia": ["abnormal lipid levels"],
    "insulin sensitivity": ["insulin responsiveness"],
    "pancreas": ["pancreatic"],
    "glucagon": ["glucagon hormone"],
    "ketoacidosis": ["dka", "diabetic ketoacidosis"],
}


def get_expansion_terms(query):
    """Return the set of extra synonym terms for a query, without repeats or terms already present."""
    query_lower = query.lower()
    extra_terms = set()

    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term in query_lower or any(s in query_lower for s in synonyms):
            extra_terms.add(term)
            extra_terms.update(synonyms)

    return {t for t in extra_terms if t not in query_lower}


