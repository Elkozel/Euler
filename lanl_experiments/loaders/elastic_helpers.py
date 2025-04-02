def get_all_records(es, index_name, query, pagination=10000):
    """
    Retrieves all records from an Elasticsearch index that match a given query, using pagination.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index.
        query (dict): The query to filter records.
        pagination (int, optional): The number of records to fetch per request. Defaults to 10000.

    Yields:
        dict: The source of each record retrieved from the index.
    """
    search_after = None  # Used for pagination to fetch the next set of records.

    while True:
        # Perform a search request with pagination and sorting.
        resp = es.search(
            index=index_name,
            query=query,
            size=pagination,
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