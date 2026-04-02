import os
import typing

from arena_models.utils.Database import Database

from . import DATABASE_NAME, AssetType, Annotation
from arena_models.impl.build import DatabaseBuilder


def query_database(
    database_path: str,
    asset_type: AssetType,
    query_target: str | dict[str, typing.Any],
    *,
    top_k_retrieve: int = 40,
    top_k_return: int = 10,
) -> Annotation:
    print(f"Querying database at {database_path} for {asset_type.value} '{query_target}'")

    db = Database(os.path.join(database_path, DATABASE_NAME))
    result = db.query(
        asset_type.value,
        query_target,
        top_k_return,
        top_k_retrieve=top_k_retrieve,
        top_k_return=top_k_return,
    )
    print(f"Query result: {result}")
    data = result['metadatas']
    if not data or not data[0]:
        raise ValueError("No results found in the database.")

    annotation_t = DatabaseBuilder.Builder(asset_type)._annotation_cls

    annotation = annotation_t.from_metadata(dict(data[0][0]))

    return annotation
