from __future__ import annotations

import logging
import math
import os
import re
import threading
import typing
import uuid
from pathlib import Path

import chromadb
import chromadb.api.models.Collection
from chromadb.utils import embedding_functions

from ..impl import Annotation

logger = logging.getLogger(__name__)

TextOrEmbedding = typing.Text | typing.List[float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENCODER_LOCK = threading.Lock()
_ENCODER_CACHE: tuple[typing.Any, typing.Any] | None = None
_ENCODER_UNAVAILABLE = False
_TEXT_EMBEDDING_CACHE: dict[str, list[float]] = {}
_METADATA_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


class Database:
    def __init__(self, path: Path, model: typing.Optional[str] = None):
        if model is None:
            model = "all-MiniLM-L6-v2"
        if model != "all-MiniLM-L6-v2":
            raise ValueError("Only all-MiniLM-L6-v2 is supported.")

        self._client = chromadb.PersistentClient(path=str(path))
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def collection(self, name) -> chromadb.api.models.Collection.Collection:
        """Get or create a ChromaDB collection."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_function,
        )

    def store(self, collection: str, annotation: Annotation):
        """Store the text embedding in a ChromaDB collection."""
        unique_id = uuid.uuid4().hex

        metadata = dict(annotation.as_metadata)

        self.collection(collection).add(
            documents=[annotation.as_text],
            metadatas=[metadata],
            ids=[unique_id]
        )

    def list_all(self, collection: str):
        """List all paths in the collection."""
        return self.collection(collection).get()

    def query(
        self,
        collection: str,
        embedding: TextOrEmbedding | dict[str, typing.Any],
        num_results: int = 1,
        *,
        top_k_retrieve: int | None = None,
        top_k_return: int | None = None,
    ):
        """Query the collection for similar embeddings.

        Backward compatible:
        - query(collection, "chair", 5)
        - query(collection, [0.1, ...], 5)

        Context-aware:
        - query(collection, context_dict, top_k_retrieve=40, top_k_return=10)
        """
        context_payload = self._normalize_context_payload(embedding)

        n_retrieve = max(1, int(top_k_retrieve if top_k_retrieve is not None else num_results))
        n_return = max(1, int(top_k_return if top_k_return is not None else num_results))

        if context_payload is not None:
            object_query = str(context_payload.get("object_description") or "").strip()
            if not object_query:
                result = self.collection(collection).query(query_texts=[""], n_results=n_return)
                result["ranking_mode"] = "text"
                return result
            base = self.collection(collection).query(
                query_texts=[object_query],
                n_results=n_retrieve,
            )
            return self._rerank_query_result(
                collection=collection,
                result=base,
                context_payload=context_payload,
                top_k_return=n_return,
            )

        if isinstance(embedding, str):
            result = self.collection(collection).query(
                query_texts=[embedding],
                n_results=n_retrieve,
            )
            result["ranking_mode"] = "text"
            return result

        if isinstance(embedding, dict):
            text = str(embedding.get("object_description") or embedding.get("query") or "").strip()
            result = self.collection(collection).query(
                query_texts=[text or ""],
                n_results=n_retrieve,
            )
            result["ranking_mode"] = "text"
            return result

        result = self.collection(collection).query(
            query_embeddings=[embedding],
            n_results=n_retrieve,
        )
        result["ranking_mode"] = "embedding"
        return result

    def query_context(
        self,
        collection: str,
        context_payload: dict[str, typing.Any],
        top_k_retrieve: int = 40,
        top_k_return: int = 10,
    ):
        return self.query(
            collection=collection,
            embedding=context_payload,
            num_results=top_k_return,
            top_k_retrieve=top_k_retrieve,
            top_k_return=top_k_return,
        )

    def get_embedding(self, text: str) -> list[float] | None:
        """Return the SigLIP embedding for *text*, or None if unavailable."""
        return self._normalized_embedding(text)

    def embedding_similarity(self, text_a: str, text_b: str) -> float | None:
        """Cosine similarity between SigLIP embeddings of two texts (0..1), or None."""
        a = self._normalized_embedding(text_a)
        b = self._normalized_embedding(text_b)
        return self._vector_similarity(a, b)

    def _to_embedding(self, value: TextOrEmbedding) -> list[float]:
        if isinstance(value, str):
            return self._embedding_function([value])[0]
        return value

    def get_distance(self, text1: TextOrEmbedding, text2: TextOrEmbedding):
        """Get distance between two vectors"""
        embedding_text_1 = self._to_embedding(text1)
        embedding_text_2 = self._to_embedding(text2)
        distance = 0
        for i, val1 in enumerate(embedding_text_1):
            val2 = embedding_text_2[i]
            distance += (val1 - val2) ** 2
        return round(distance / 100, 3)

    @staticmethod
    def _safe_float(value: typing.Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_context_payload(self, payload: typing.Any) -> dict[str, typing.Any] | None:
        if not isinstance(payload, dict):
            return None
        if "object_description" not in payload:
            return None

        neighbours = payload.get("neighbouring_assets") or []
        if not isinstance(neighbours, list):
            neighbours = [neighbours]

        bbox = payload.get("target_bbox")
        if isinstance(bbox, dict):
            bbox_payload = {
                "min_x": self._safe_float(bbox.get("min_x")),
                "max_x": self._safe_float(bbox.get("max_x")),
                "min_y": self._safe_float(bbox.get("min_y")),
                "max_y": self._safe_float(bbox.get("max_y")),
                "min_z": self._safe_float(bbox.get("min_z")),
                "max_z": self._safe_float(bbox.get("max_z")),
            }
        else:
            bbox_payload = None

        return {
            "object_description": str(payload.get("object_description") or ""),
            "room_prompt": str(payload.get("room_prompt") or ""),
            "neighbouring_assets": [str(item) for item in neighbours if str(item).strip()],
            "target_bbox": bbox_payload,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_RE.findall((text or "").lower()))

    def _text_overlap(self, reference: str, candidate: str, neutral_if_empty: bool = False) -> float:
        left = self._tokenize(reference)
        right = self._tokenize(candidate)
        if not left or not right:
            return 0.5 if neutral_if_empty else 0.0
        return len(left & right) / max(1, len(left))

    def _load_encoder(self) -> tuple[typing.Any, typing.Any] | None:
        global _ENCODER_CACHE, _ENCODER_UNAVAILABLE
        with _ENCODER_LOCK:
            if _ENCODER_CACHE is not None:
                return _ENCODER_CACHE
            if _ENCODER_UNAVAILABLE:
                return None

        model_name = os.getenv("ARENA_MODELS_SIGLIP_TEXT_MODEL", "google/siglip-base-patch16-224")
        local_only = os.getenv("ARENA_MODELS_EMBEDDING_LOCAL_ONLY", "0").strip().lower() in ("1", "true", "yes")

        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            logger.warning("SigLIP encoder unavailable: 'transformers' package not installed.")
            with _ENCODER_LOCK:
                _ENCODER_UNAVAILABLE = True
            return None

        try:
            logger.info("Loading SigLIP encoder '%s' (local_only=%s)...", model_name, local_only)
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
            model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
            model.eval()
            logger.info("SigLIP encoder loaded successfully.")
        except Exception as error:
            logger.warning("SigLIP encoder unavailable: failed to load '%s': %s", model_name, error)
            with _ENCODER_LOCK:
                _ENCODER_UNAVAILABLE = True
            return None

        with _ENCODER_LOCK:
            _ENCODER_CACHE = (tokenizer, model)
        return tokenizer, model

    def _normalized_embedding(self, text: str) -> list[float] | None:
        global _ENCODER_UNAVAILABLE
        content = " ".join((text or "").split()).strip()
        if not content:
            return None

        key = content.lower()
        with _ENCODER_LOCK:
            cached = _TEXT_EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached

        encoder = self._load_encoder()
        if encoder is None:
            return None
        tokenizer, model = encoder

        try:
            import torch
        except Exception:
            logger.warning("SigLIP encoder unavailable: 'torch' package not installed.")
            with _ENCODER_LOCK:
                _ENCODER_UNAVAILABLE = True
            return None

        try:
            tokens = tokenizer(
                content,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                output = model(**tokens)

            if hasattr(output, "text_embeds") and output.text_embeds is not None:
                vec = output.text_embeds[0]
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                vec = output.pooler_output[0]
            elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                hidden = output.last_hidden_state[0]
                mask = tokens.get("attention_mask")
                if mask is None:
                    vec = hidden.mean(dim=0)
                else:
                    weights = mask[0].unsqueeze(-1).to(hidden.dtype)
                    denom = weights.sum().clamp(min=1e-9)
                    vec = (hidden * weights).sum(dim=0) / denom
            else:
                logger.warning("SigLIP model output has no usable embedding attribute.")
                return None

            norm = torch.linalg.norm(vec).item()
            if not math.isfinite(norm) or norm <= 0.0:
                return None
            normalized = (vec / norm).detach().cpu().to(torch.float32).tolist()
        except Exception as error:
            logger.warning("SigLIP embedding inference failed: %s", error)
            return None

        with _ENCODER_LOCK:
            _TEXT_EMBEDDING_CACHE[key] = normalized
        return normalized

    @staticmethod
    def _vector_similarity(left: list[float] | None, right: list[float] | None) -> float | None:
        if left is None or right is None:
            return None
        if len(left) != len(right):
            return None
        dot = sum(a * b for a, b in zip(left, right))
        return max(0.0, min(1.0, (dot + 1.0) * 0.5))

    @staticmethod
    def _candidate_text(metadata: dict[str, typing.Any]) -> str:
        return " ".join(
            str(metadata.get(key, "")).strip()
            for key in ("name", "desc", "note", "tags", "material", "color", "hoi", "face")
            if metadata.get(key) is not None
        ).strip()

    @staticmethod
    def _context_text(room_prompt: str, neighbours: list[str]) -> str:
        neighbours_text = ", ".join(item for item in neighbours if item.strip())
        parts: list[str] = []
        if room_prompt.strip():
            parts.append(room_prompt.strip())
        if neighbours_text:
            parts.append(f"Nearby assets: {neighbours_text}")
        return " | ".join(parts).strip()

    def _metadata_embedding(
        self,
        collection_name: str,
        metadata: dict[str, typing.Any],
        candidate_text: str,
    ) -> list[float] | None:
        cache_key_part = str(metadata.get("path") or metadata.get("name") or candidate_text).strip()
        cache_key = (collection_name, cache_key_part)
        with _ENCODER_LOCK:
            cached = _METADATA_EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            return cached

        encoded = self._normalized_embedding(candidate_text)
        if encoded is None:
            return None
        with _ENCODER_LOCK:
            _METADATA_EMBEDDING_CACHE[cache_key] = encoded
        return encoded

    def _embedding_similarity(
        self,
        query_text: str,
        candidate_text: str,
        collection_name: str,
        metadata: dict[str, typing.Any],
    ) -> float | None:
        query_embedding = self._normalized_embedding(query_text)
        candidate_embedding = self._metadata_embedding(collection_name, metadata, candidate_text)
        return self._vector_similarity(query_embedding, candidate_embedding)

    def _parse_asset_bbox(self, value: typing.Any) -> tuple[float, float, float] | None:
        if isinstance(value, dict):
            x = self._safe_float(value.get("x") or value.get("width"), float("nan"))
            y = self._safe_float(value.get("y") or value.get("depth"), float("nan"))
            z = self._safe_float(value.get("z") or value.get("height") or 0.0, float("nan"))
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                return max(0.01, x), max(0.01, y), max(0.0, z)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x = self._safe_float(value[0], float("nan"))
            y = self._safe_float(value[1], float("nan"))
            z = self._safe_float(value[2] if len(value) > 2 else 0.0, float("nan"))
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                return max(0.01, x), max(0.01, y), max(0.0, z)
        if isinstance(value, str):
            parts = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value)
            if len(parts) >= 2:
                x = self._safe_float(parts[0], 0.01)
                y = self._safe_float(parts[1], 0.01)
                z = self._safe_float(parts[2], 0.0) if len(parts) > 2 else 0.0
                return max(0.01, x), max(0.01, y), max(0.0, z)
        return None

    def _bbox_fit(self, target_bbox: dict[str, float] | None, metadata: dict[str, typing.Any]) -> float:
        if not target_bbox:
            return 0.5
        asset_bbox = self._parse_asset_bbox(metadata.get("bounding_box"))
        if asset_bbox is None:
            return 0.5

        target_x = max(0.01, abs(self._safe_float(target_bbox.get("max_x")) - self._safe_float(target_bbox.get("min_x"))))
        target_y = max(0.01, abs(self._safe_float(target_bbox.get("max_y")) - self._safe_float(target_bbox.get("min_y"))))
        target_z = max(0.0, abs(self._safe_float(target_bbox.get("max_z")) - self._safe_float(target_bbox.get("min_z"))))
        asset_x, asset_y, asset_z = asset_bbox

        fit_x = min(target_x, asset_x) / max(target_x, asset_x)
        fit_y = min(target_y, asset_y) / max(target_y, asset_y)
        if target_z <= 0.0 or asset_z <= 0.0:
            return (fit_x + fit_y) / 2.0
        fit_z = min(target_z, asset_z) / max(target_z, asset_z)
        return (fit_x + fit_y + fit_z) / 3.0

    @staticmethod
    def _rebuild_result(original: dict[str, typing.Any], rows: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        keys = list(original.keys())
        rebuilt: dict[str, typing.Any] = {}
        for key in keys:
            rebuilt[key] = [[]]

        scores: list[float] = []
        for row in rows:
            scores.append(float(row.get("score", 0.0)))
            for key in keys:
                values = original.get(key)
                if isinstance(values, list) and values and isinstance(values[0], list):
                    idx = int(row["index"])
                    if idx < len(values[0]):
                        rebuilt[key][0].append(values[0][idx])
        rebuilt["scores"] = [scores]
        return rebuilt

    @staticmethod
    def _retrieval_similarity(row: dict[str, typing.Any]) -> float | None:
        score = row.get("score")
        if isinstance(score, (int, float)):
            value = float(score)
            if 0.0 <= value <= 1.0:
                return value
            return 1.0 / (1.0 + math.exp(-value))
        distance = row.get("distance")
        if isinstance(distance, (int, float)):
            return 1.0 / (1.0 + max(0.0, float(distance)))
        return None

    def _rerank_query_result(
        self,
        collection: str,
        result: dict[str, typing.Any],
        context_payload: dict[str, typing.Any],
        top_k_return: int,
    ) -> dict[str, typing.Any]:
        metadatas = result.get("metadatas")
        if not isinstance(metadatas, list) or not metadatas or not isinstance(metadatas[0], list):
            return result

        metadata_rows = metadatas[0]
        distances = result.get("distances")
        distance_rows: list[typing.Any] = []
        if isinstance(distances, list) and distances and isinstance(distances[0], list):
            distance_rows = list(distances[0])

        object_query = str(context_payload.get("object_description") or "")
        room_prompt = str(context_payload.get("room_prompt") or "")
        neighbours = context_payload.get("neighbouring_assets") or []
        if not isinstance(neighbours, list):
            neighbours = [neighbours]
        context_query = self._context_text(room_prompt, [str(item) for item in neighbours])
        target_bbox = context_payload.get("target_bbox")
        if not isinstance(target_bbox, dict):
            target_bbox = None

        used_siglip = False
        rows: list[dict[str, typing.Any]] = []
        for idx, metadata in enumerate(metadata_rows):
            if not isinstance(metadata, dict):
                continue

            row: dict[str, typing.Any] = {"index": idx, "metadata": metadata}
            if idx < len(distance_rows):
                row["distance"] = distance_rows[idx]

            candidate_text = self._candidate_text(metadata)
            retrieval_sim = self._retrieval_similarity(row)

            object_embed_sim = self._embedding_similarity(
                query_text=object_query,
                candidate_text=candidate_text,
                collection_name=collection,
                metadata=metadata,
            )
            context_embed_sim = self._embedding_similarity(
                query_text=context_query,
                candidate_text=candidate_text,
                collection_name=collection,
                metadata=metadata,
            )

            object_lexical_sim = self._text_overlap(object_query, candidate_text)
            context_lexical_sim = self._text_overlap(context_query, candidate_text, neutral_if_empty=True)
            bbox_fit = self._bbox_fit(target_bbox, metadata)

            object_sim = object_embed_sim
            if object_sim is None:
                object_sim = retrieval_sim if retrieval_sim is not None else object_lexical_sim
            else:
                used_siglip = True

            context_sim = context_embed_sim if context_embed_sim is not None else context_lexical_sim
            if context_embed_sim is not None:
                used_siglip = True

            score = (0.55 * object_sim) + (0.30 * context_sim) + (0.15 * bbox_fit)
            row["score"] = score
            rows.append(row)

        ranking_mode = "siglip" if used_siglip else "lexical"
        logger.info(
            "Reranking %d candidates for '%s' using %s scoring.",
            len(rows), object_query, ranking_mode,
        )

        rows.sort(
            key=lambda row: (
                -float(row.get("score", 0.0)),
                str((row.get("metadata") or {}).get("name", "")),
            )
        )
        rebuilt = self._rebuild_result(result, rows[:top_k_return])
        rebuilt["ranking_mode"] = ranking_mode
        return rebuilt
