import json
import os
import re
import time
import pandas as pd
import requests

# Load the filtering keywords
with open("./config/search_keywords.json", "r") as file:
    data = json.load(file)

SEARCH_QUERY = data["search_query_keywords"]


# Define functions used for checkpointing
def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return {"cursor": "*", "page": 0}

    with open(checkpoint_path, "r") as file:
        checkpoint = json.load(file)

    return {
        "cursor": checkpoint.get("cursor", "*"),
        "page": checkpoint.get("page", 0),
    }


def save_checkpoint(checkpoint_path, cursor, page):
    with open(checkpoint_path, "w") as file:
        json.dump({"cursor": cursor, "page": page}, file, indent=2)


def remove_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def load_existing_ids(output_csv):
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()

    try:
        df = pd.read_csv(output_csv, usecols=["openalex_id"])
    except ValueError:
        return set()

    return set(df["openalex_id"].dropna().astype(str))


def split_boolean_or_groups(search_query):
    groups = re.findall(r"\((.*?)\)", search_query)
    parsed_groups = []

    for group in groups:
        terms = []
        for term in re.split(r"\s+OR\s+", group):
            cleaned = term.strip().strip('"').strip()
            if cleaned:
                terms.append(cleaned)
        if terms:
            parsed_groups.append(terms)

    return parsed_groups


def build_legacy_title_abstract_filters(search_query):
    groups = split_boolean_or_groups(search_query)
    if not groups:
        return None

    clauses = []
    for group in groups:
        escaped_terms = []
        for term in group:
            if " " in term:
                escaped_terms.append(f'"{term}"')
            else:
                escaped_terms.append(term)
        clauses.append(f"title_and_abstract.search:{'|'.join(escaped_terms)}")

    return clauses


def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""

    positions = []
    for word, indexes in abstract_inverted_index.items():
        for index in indexes or []:
            positions.append((index, word))

    if not positions:
        return ""

    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def safe_join(values):
    cleaned = []
    seen = set()

    for value in values or []:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    return "; ".join(cleaned)


def extract_authors(authorships):
    return safe_join(authorship.get("author", {}).get("display_name") for authorship in authorships)


def extract_institutions(authorships):
    institutions = []
    for authorship in authorships:
        for institution in authorship.get("institutions", []):
            institutions.append(institution.get("display_name"))

    return safe_join(institutions)


def extract_institution_countries(authorships):
    countries = []
    for authorship in authorships:
        for country in authorship.get("countries", []) or []:
            countries.append(country)
        for institution in authorship.get("institutions", []) or []:
            countries.append(institution.get("country_code"))

    return safe_join(countries)


def extract_concepts(concepts):
    return safe_join(concept.get("display_name") for concept in concepts)


def extract_mesh_terms(mesh):
    return safe_join(term.get("descriptor_name") for term in mesh)


def extract_keywords(keywords):
    return safe_join(keyword.get("display_name") for keyword in keywords)


def get_primary_location_data(primary_location):
    source = (primary_location or {}).get("source") or {}
    return {
        "source_name": source.get("display_name", ""),
        "source_type": source.get("type", ""),
        "source_issn_l": source.get("issn_l", ""),
    }


def build_row(paper):
    authorships = paper.get("authorships", [])
    location_data = get_primary_location_data(paper.get("primary_location"))
    biblio = paper.get("biblio") or {}

    return {
        "openalex_id": paper.get("id", ""),
        "title": paper.get("display_name", ""),
        "publication_date": paper.get("publication_date", ""),
        "publication_year": paper.get("publication_year"),
        "cited_by_count": paper.get("cited_by_count"),
        "authors_count": len(authorships),
        "authors": extract_authors(authorships),
        "institutions": extract_institutions(authorships),
        "institution_countries": extract_institution_countries(authorships),
        "abstract": reconstruct_abstract(paper.get("abstract_inverted_index")),
        "doi": paper.get("doi", ""),
        "type": paper.get("type", ""),
        "language": paper.get("language", ""),
        "is_open_access": (paper.get("open_access") or {}).get("is_oa"),
        "source_name": location_data["source_name"],
        "source_type": location_data["source_type"],
        "source_issn_l": location_data["source_issn_l"],
        "volume": biblio.get("volume", ""),
        "issue": biblio.get("issue", ""),
        "first_page": biblio.get("first_page", ""),
        "last_page": biblio.get("last_page", ""),
        "concepts": extract_concepts(paper.get("concepts")),
        "mesh_terms": extract_mesh_terms(paper.get("mesh")),
        "keywords": extract_keywords(paper.get("keywords")),
        "referenced_works_count": paper.get("referenced_works_count"),
    }


def append_rows_to_csv(rows, output_csv):
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    df.to_csv(output_csv, mode="a", header=not file_exists, index=False)
    return len(df)


def raise_detailed_http_error(response):
    try:
        payload = response.json()
        detail = payload.get("message") or payload.get("error") or json.dumps(payload)
    except ValueError:
        detail = response.text.strip() or "No response body"

    raise requests.HTTPError(
        f"{response.status_code} {response.reason}: {detail}",
        response=response,
    )


def fetch_page(url, params, fallback_filter_clauses=None, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"Request failed ({exc}). Retrying in {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
            continue

        if response.ok:
            return response.json()

        # There are very often server errors (error 500), retry on these errors
        if response.status_code >= 500 and attempt < max_retries:
            wait = 2 ** attempt
            print(f"Server error {response.status_code}. Retrying in {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
            continue

        if response.status_code == 400 and fallback_filter_clauses:
            retry_params = {key: value for key, value in params.items() if key != "search"}
            existing_filter = retry_params.get("filter", "")
            combined_filters = [existing_filter] if existing_filter else []
            combined_filters.extend(fallback_filter_clauses)
            retry_params["filter"] = ",".join(combined_filters)

            retry_response = requests.get(url, params=retry_params, timeout=60)
            if retry_response.ok:
                print("Primary search query was rejected; switched to legacy title/abstract filters.")
                return retry_response.json()

            raise_detailed_http_error(retry_response)

        raise_detailed_http_error(response)


def fetch_publications(
    search_query=SEARCH_QUERY,
    start_year=1980,
    end_year=2026,
    output_csv="openalex_papers_detailed.csv",
    checkpoint_path="openalex_papers_detailed.checkpoint.json",
    max_pages=None,
    api_key=None,
):
    url = "https://api.openalex.org/works"
    checkpoint = load_checkpoint(checkpoint_path)
    cursor = checkpoint["cursor"]
    page = checkpoint["page"]
    total_saved = 0
    existing_ids = load_existing_ids(output_csv)
    total_saved += len(existing_ids)
    fallback_filter_clauses = build_legacy_title_abstract_filters(search_query)

    params_base = {
        "search": search_query,
        "filter": f"from_publication_date:{start_year}-01-01,to_publication_date:{end_year}-12-31",
        "select": ",".join(
            [
                "id",
                "display_name",
                "publication_date",
                "publication_year",
                "cited_by_count",
                "authorships",
                "abstract_inverted_index",
                "doi",
                "type",
                "language",
                "open_access",
                "primary_location",
                "biblio",
                "concepts",
                "mesh",
                "keywords",
                "referenced_works_count",
            ]
        ),
        "per_page": 100,
    }
    if api_key:
        params_base["api_key"] = api_key

    print(
        f"Starting from page {page + 1} with cursor '{cursor}'. "
        f"Already saved papers: {len(existing_ids)}"
    )

    try:
        while True:
            if max_pages is not None and page >= max_pages:
                break

            params = dict(params_base)
            params["cursor"] = cursor

            data = fetch_page(url, params, fallback_filter_clauses=fallback_filter_clauses)
            results = data.get("results", [])

            if not results:
                break

            rows_to_append = []
            for paper in results:
                openalex_id = str(paper.get("id", "")).strip()
                if not openalex_id or openalex_id in existing_ids:
                    continue

                rows_to_append.append(build_row(paper))
                existing_ids.add(openalex_id)

            saved_now = append_rows_to_csv(rows_to_append, output_csv)
            total_saved += saved_now

            page += 1
            next_cursor = data.get("meta", {}).get("next_cursor")
            save_checkpoint(checkpoint_path, next_cursor, page)

            print(
                f"Fetched page {page}, received {len(results)} papers, "
                f"saved {saved_now} new rows, total saved: {total_saved}"
            )

            if not next_cursor:
                remove_checkpoint(checkpoint_path)
                break

            cursor = next_cursor
            time.sleep(0.5)

    except KeyboardInterrupt:
        save_checkpoint(checkpoint_path, cursor, page)
        raise

    print(f"Finished. Dataset saved to {output_csv}")


if __name__ == "__main__":
    fetch_publications(
        search_query=SEARCH_QUERY,
        start_year=1980,
        end_year=2026,
        output_csv="openalex_ai_medicine_detailed.csv",
        checkpoint_path="openalex_ai_medicine_detailed.checkpoint.json",
        max_pages=None,
        api_key=os.getenv("OPENALEX_API_KEY"),
    )
