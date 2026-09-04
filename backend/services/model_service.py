import pickle
import sys
from typing import Any

from fastapi import HTTPException

from config import MODEL_PATH
from cinematch_model import CineMatchSVD, Prediction


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "CineMatchSVD":
            sys.modules.setdefault("__main__", sys.modules["__main__"])
            setattr(sys.modules["__main__"], "CineMatchSVD", CineMatchSVD)
            setattr(sys.modules["__main__"], "Prediction", Prediction)
            return CineMatchSVD
        if module == "__main__" and name == "Prediction":
            sys.modules.setdefault("__main__", sys.modules["__main__"])
            setattr(sys.modules["__main__"], "Prediction", Prediction)
            return Prediction
        return super().find_class(module, name)


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.initialized = False

    def initialize(self) -> None:
        if self.initialized:
            return

        try:
            with MODEL_PATH.open("rb") as handle:
                self.model = _CompatUnpickler(handle).load()
            self.initialized = True
        except AttributeError:
            try:
                with MODEL_PATH.open("rb") as handle:
                    self.model = pickle.load(handle)
                self.initialized = True
            except Exception as exc:  # pragma: no cover - defensive fallback
                raise HTTPException(status_code=500, detail="Unable to load the trained SVD model.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="SVD model file not found.") from exc
        except Exception as exc:  # pragma: no cover - model corruption fallback
            raise HTTPException(status_code=500, detail="Unable to load the trained SVD model.") from exc

    def get_model(self) -> Any:
        if not self.initialized or self.model is None:
            self.initialize()
        return self.model


model_service = ModelService()
