from typing import Any, Dict, List, Optional, Union

from qdrant_client.http import models

from qdrant_store.client import get_client


def upsert_point(collection: str, point_id: str, vectors: Dict[str, List[float]], payload: Dict[str, Any]) -> None:
    client = get_client()
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(id=point_id, vector=vectors, payload=payload)],
    )


def update_payload(collection: str, point_id: str, payload: Dict[str, Any]) -> None:
    client = get_client()
    client.set_payload(collection_name=collection, payload=payload, points=[point_id])


def get_point(collection: str, point_id: str):
    client = get_client()
    result = client.retrieve(collection_name=collection, ids=[point_id], with_payload=True)
    return result[0] if result else None


def search_vectors(
    collection: str,
    vector_name: str,
    vector: List[float],
    limit: int = 5,
    filters: Optional[Union[models.Filter, Dict[str, Any]]] = None,
) -> List[models.ScoredPoint]:
    client = get_client()
    if filters is not None and not isinstance(filters, models.Filter):
        if hasattr(models.Filter, "model_validate"):
            filters = models.Filter.model_validate(filters)
        else:
            filters = models.Filter.parse_obj(filters)
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )
        return response.points
    return client.search(
        collection_name=collection,
        query_vector=(vector_name, vector),
        limit=limit,
        query_filter=filters,
        with_payload=True,
    )


def scroll_points(collection: str, limit: int = 100, offset: Optional[int] = None):
    client = get_client()
    return client.scroll(collection_name=collection, limit=limit, offset=offset, with_payload=True)
