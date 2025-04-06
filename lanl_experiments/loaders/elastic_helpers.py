# Elasticsearch complains about the connection not being secure
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ElasticDataFetcher():
    """
    A generator-like class that retrieves all records from an Elasticsearch index
    and supports the len() function to get the total number of matching records.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index.
        query (dict): The query to filter records.
        pagination (int, optional): The number of records to fetch per request. Defaults to 10000.
    """
    def __init__(self, es, index_name, query, pagination=10000):
        self.es = es
        self.index_name = index_name
        self.query = query
        self.pagination = pagination
        self.matchcount = None # The count of matching records
        self._total_count = None  # Cache the total count of matching records

    def __iter__(self):
        search_after = None  # Used for pagination to fetch the next set of records.

        while True:
            # Perform a search request with pagination and sorting.
            resp = self.es.search(
                index=self.index_name,
                query=self.query,
                size=self.pagination,
                search_after=search_after,
                sort=[
                    {"datetime": "asc"},  # Sort by datetime in ascending order.
                    {"__id": "asc"}       # Secondary sort by __id in ascending order.
                ],
            )

            # If no records are returned, exit the loop.
            if len(resp.body["hits"]["hits"]) == 0:
                return

            # Update the search_after value for the next request.
            search_after = resp.body["hits"]["hits"][-1]["sort"]

            # Yield each record's source data.
            for record in resp.body["hits"]["hits"]:
                yield record["_source"]
    
    def __len__(self):
        # Fetch the length if not previously fetched
        if self.matchcount == None:
            resp = self.es.count(
                index=self.index_name,
                query=self.query
            )
            self.matchcount = resp.body["count"]
        
        return self.matchcount

def generate_node_map(es, index, field, query=None):
    """
    Generates a mapping of unique field values to unique IDs from an Elasticsearch index.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        index (str): The name of the Elasticsearch index.
        field (str): The field to aggregate unique values from.
        query (dict, optional): An optional query to filter records. Defaults to None.

    Returns:
        dict: A dictionary mapping unique field values to unique IDs.
    """
    # Perform an aggregation query to get unique values of the specified field.
    resp = es.search(
        index=index,
        size=0,  # No documents are returned, only aggregation results.
        query=query,
        aggs={
            "uniq": {
                "terms": {
                    "field": field,  # Aggregate unique values of the specified field.
                    "size": 100000,    # Maximum number of unique values to retrieve.
                    "order": {
                        "_key": "asc"
                    }
                }
            }
        },
    )

    # Extract the buckets from the aggregation results.
    buckets = resp.body["aggregations"]["uniq"]["buckets"]

    # Create a mapping of unique field values to unique IDs.
    nodemap = [bucket["key"] for bucket in buckets]
    return nodemap

def make_fielddata(es, index, field):
    """
    Enables fielddata for a given text field in an Elasticsearch index.

    Fielddata is required for certain operations, such as sorting or aggregations, on text fields.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        index (str): The name of the Elasticsearch index.
        field (str): The name of the field to enable fielddata for.

    Returns:
        None
    """
    # Update the index mapping to enable fielddata for the specified field.
    es.indices.put_mapping(
        index=index,
        body={
            "properties": {
                field: {
                    "type": "text",       # Ensure the field is of type "text".
                    "fielddata": True     # Enable fielddata for the field.
                }
            }
        }
    )