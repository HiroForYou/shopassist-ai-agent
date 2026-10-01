import warnings

import langchain_core  # noqa: F401  # al importarse reactiva sus warnings; el filtro debe ir despues

# Warning interno de langgraph 1.0 al importar el checkpointer; no depende del proyecto.
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")
