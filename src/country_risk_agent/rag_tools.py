from . import helpers
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import random
import requests

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

DATA = {
    '2019': helpers.load_json("data_2019.json"),
    '2021': helpers.load_json("data_2021.json"),
    '2023': helpers.load_json("data_2023.json"),
    'WID': helpers.load_csv("wdi_mortality_data.csv")
        }

# =====================================================
# Worry/Experience Gap Tool
# =====================================================

def get_worry_experience_gap(country: str, year: str):
    """
    For a given country and year, obtain the mean gap between
    how much people worry about different risks in their lives
    and how much they experience these risks. Positive numbers imply that 
    people in a country tend to worry more than they should,
    and negative numbers mean that people in a country tend
    to worry less than they should.

    Returns a summary statistic ONLY.
    This does NOT include supporting data or raw survey responses.
    Use database_query_* tools if raw data is needed.

    Args: 
        country: string specifying a country. 
        year: string specifying a year. 
    
    Returns: 
        A dictionary contain the mean gap between worry and experience 
        in the requested country for that year, as well as the mean for 
        all countries for comparison.

    """
    if year not in ['2019', '2021', '2023']:
        return "Year must be one of 2019, 2021, or 2023."
    
    data = DATA[year]

    country_vals = [
    d['metadata']['quantitative_data'].get("Worry and experience gap")
    for d in data
    if d['metadata']['demographics']['Country Name'] == country
    and d['metadata']['quantitative_data'].get("Worry and experience gap") is not None
    ]

    all_vals = [
        d['metadata']['quantitative_data'].get("Worry and experience gap")
        for d in data
        if d['metadata']['quantitative_data'].get("Worry and experience gap") is not None
    ]

    mean_gap_country = helpers.safe_mean(country_vals)
    mean_gap_all = helpers.safe_mean(all_vals)

    return str({
        "Country": country,
        "Year": year,
        "Country-Specific Mean Gap": mean_gap_country,
        "Mean Gap Across All Countries": mean_gap_all
    })
    
# =====================================================
# Wikipedia Summary Tool
# =====================================================

def get_country_wiki_summary(country: str) -> str:
    """
    Return the introductory summary from a country's Wikipedia page. Try
    multiple variations on a country's name if the first attempt does not
    yield a result.

    Args:
        country: Country name (e.g., 'France', 'Japan', 'United States')

    Returns:
        Summary string or error message.
    """
    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
        f"{country.replace(' ', '_')}"
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "CountrySummaryBot/1.0"},
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        if "extract" in data:
            return data["extract"]

        return f"No summary found for '{country}'."

    except requests.RequestException as e:
        return f"Wikipedia request failed: {e}"


# =====================================================
# 2019 Data Tools
# =====================================================

def country_search_2019(query:str, k:int):
    """
    Search the 2019 data for countries that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find countries relevant
        to the prompt.
        k: the number of relevant countries to return.

    Returns:
        A list of the k countries whose embeddings are closest to
        the embedding of the query.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2019/country_db_2019",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results



def field_search_2019(query:str, k:int):
    """
    Search the 2019 data for data fields that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find data fields relevant
        to the prompt.
        k: the number of relevant fields to return.

    Returns:
        A list of the k data fields whose embeddings are closest to
        the embedding of the query.
    """
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2019/column_db_2019",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results


def database_query_2019(countries:list[str], fields:list[str],
                        max_records: int = 50):
    """
    Search the 2019 data for survey responses from people in the input
    countries, filtered to just provide respondant metadata and their 
    responses to the questions specified by the input fields.

    Args:
        countries: a list of strings specifying the countries that data
        is requested from.
        fields: a list of strings specifying the data fields of interest.

    Returns:
        A list of dictionaries with participant metadata from the relevant
        countries, plus their responses to the questions specified by the input fields.
    """
    data_2019 = DATA['2019']

    country_filter = [
        d for d in data_2019
        if d.get('metadata', {})
            .get('demographics', {})
            .get('Country Name') in countries
    ]
    
    
    random.seed(42)

    sample = random.sample(
        country_filter,
        k=min(max_records, len(country_filter))
    )

    results = []

    for c in sample:
        response_dict = {
            "metadata": {
                "demographics": c['metadata']['demographics'],
                "quantitative_data": c['metadata']['quantitative_data']
            },
            "qualitative_data": {}
        }
        for field in fields:
            response_dict['qualitative_data'][field] = (
                c.get('qualitative_data', {}).get(field)
            )
        results.append(response_dict)

    return helpers.clean_nan(results)

# =====================================================
# 2021 Data Tools
# =====================================================

def country_search_2021(query:str, k:int):
    """
    Search the 2021 data for countries that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find countries relevant
        to the prompt.
        k: the number of relevant countries to return.

    Returns:
        A list of the k countries whose embeddings are closest to
        the embedding of the query.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2021/country_db_2021",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results

def field_search_2021(query:str, k:int):
    """
    Search the 2021 data for data fields that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find data fields relevant
        to the prompt.
        k: the number of relevant fields to return.

    Returns:
        A list of the k data fields whose embeddings are closest to
        the embedding of the query.
    """
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2021/column_db_2021",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results


def database_query_2021(countries:list[str], fields:list[str],
                        max_records: int = 50):
    """
    Search the 2021 data for survey responses from people in the input
    countries, filtered to just provide respondant metadata and their 
    responses to the questions specified by the input fields.

    Args:
        countries: a list of strings specifying the countries that data
        is requested from.
        fields: a list of strings specifying the data fields of interest.

    Returns:
        A list of dictionaries with participant metadata from the relevant
        countries, plus their responses to the questions specified by the input fields.
    """
    data_2021 = DATA['2021']

    country_filter = [
        d for d in data_2021
        if d.get('metadata', {})
            .get('demographics', {})
            .get('Country Name') in countries
    ]

    random.seed(42)

    sample = random.sample(
        country_filter,
        k=min(max_records, len(country_filter))
    )

    results = []

    for c in sample:
        response_dict = {
            "metadata": {
                "demographics": c['metadata']['demographics'],
                "quantitative_data": c['metadata']['quantitative_data']
            },
            "qualitative_data": {}
        }
        for field in fields:
            response_dict['qualitative_data'][field] = (
                c.get('qualitative_data', {}).get(field)
            )
        results.append(response_dict)

    return helpers.clean_nan(results)

# =====================================================
# 2023 Data Tools
# =====================================================

def country_search_2023(query:str, k:int):
    """
    Search the 2023 data for countries that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find countries relevant
        to the prompt.
        k: the number of relevant countries to return. 

    Returns:
        A list of the k countries whose embeddings are closest to
        the embedding of the query.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2023/country_db_2023",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results

def field_search_2023(query:str, k:int):
    """
    Search the 2023 data for data fields that are relevant to the
    user prompt using k-nearest neighbors semantic search. 

    Args:
        query: a linguistic query designed to find data fields relevant
        to the prompt.
        k: the number of relevant fields to return.

    Returns:
        A list of the k data fields whose embeddings are closest to
        the embedding of the query.
    """
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/_2023/column_db_2023",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results


def database_query_2023(countries:list[str], fields:list[str],
                        max_records: int = 50):
    """
    Search the 2023 data for survey responses from people in the input
    countries, filtered to just provide respondant metadata and their 
    responses to the questions specified by the input fields.

    Args:
        countries: a list of strings specifying the countries that data
        is requested from.
        fields: a list of strings specifying the data fields of interest.

    Returns:
        A list of dictionaries with participant metadata from the relevant
        countries, plus their responses to the questions specified by the input fields.
    """
    data_2023 = DATA['2023']

    country_filter = [
        d for d in data_2023
        if d.get('metadata', {})
            .get('demographics', {})
            .get('Country Name') in countries
    ]

    random.seed(42)

    sample = random.sample(
        country_filter,
        k=min(max_records, len(country_filter))
    )

    results = []

    for c in sample:
        response_dict = {
            "metadata": {
                "demographics": c['metadata']['demographics'],
                "quantitative_data": c['metadata']['quantitative_data']
            },
            "qualitative_data": {}
        }
        for field in fields:
            response_dict['qualitative_data'][field] = (
                c.get('qualitative_data', {}).get(field)
            )
        results.append(response_dict)

    return helpers.clean_nan(results)


# =====================================================
# Mortality Data Tools
# =====================================================

def country_search_mortality(query:str, k:int):
    """
    Search the world development index's mortality data for countries 
    that are relevant to the user prompt using k-nearest neighbors semantic 
    search. 

    Args:
        query: a linguistic query designed to find countries relevant
        to the prompt.
        k: the number of relevant countries to return.

    Returns:
        A list of the k countries whose embeddings are closest to
        the embedding of the query.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )
    vectorstore = Chroma(
    persist_directory="./src/vector_stores/wdi/country_db_wdi",
    embedding_function=embeddings
    )

    results = vectorstore.similarity_search(
    query,
    k=k
    )
    return results

def get_mortality_data(country:str, year:str):
    """
    Obtain world development index mortality data for a given country
    in a given year.

    Args:
        country: the country to obtain data for.
        year: the year to obtain data for.

    Returns:
        A list of the k countries whose embeddings are closest to
        the embedding of the query.
    """
    data = DATA['WID']

    data = data.dropna()

    data_filtered1 = data[data['Country Name'] == country]
    
    results = ""
    for _, row in data_filtered1.iterrows():
        results += f"{row['Series Name']}: {row[f'{year} [YR{year}]']}\n"

    return results

