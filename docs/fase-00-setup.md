# Fase 0: setup

Entorno base: Ollama con los modelos, tool calling verificado, Qdrant y trazas en LangSmith.

## Servicios

| Servicio | Rol | Puerto |
|---|---|---|
| `ollama` | Servidor de modelos; volumen `ollama_data` | 11434 |
| `ollama-pull` | Descarga `OLLAMA_MODEL`, `OLLAMA_FALLBACK_MODEL` y `OLLAMA_EMBED_MODEL`. Perfil `tools` (solo con `docker compose run`) | - |
| `qdrant` | Base vectorial de la Fase 3; volumen `qdrant_data` | 6333 |
| `app` | Contenedor de trabajo para scripts y tests; monta el proyecto en `/app` (los cambios de código no requieren rebuild) | - |
| `api` | API de la Fase 7 | 8000 |
| `prometheus`, `grafana` | Monitoreo de la Fase 7; perfil `monitoring` | 9090, 3000 |
| LangSmith | SaaS (plan gratuito); la versión self-hosted requiere licencia enterprise | - |

`ollama-pull` no es dependencia de `app`: un `ollama pull` puede cambiar los pesos de un tag y romper la
comparabilidad entre experimentos (ver [Fase 5](fase-05-observabilidad.md#hallazgos)).

## Instalación

1. Docker Desktop con backend WSL2. Límite de recursos de WSL2 en `C:\Users\<usuario>\.wslconfig`, aplicado con
   `wsl --shutdown`:
   ```ini
   [wsl2]
   memory=8GB
   processors=6
   swap=4GB

   [experimental]
   autoMemoryReclaim=gradual
   ```
2. API key de LangSmith (smith.langchain.com → Settings → API Keys).
3. Configuración:
   ```powershell
   Copy-Item .env.example .env      # completar LANGSMITH_API_KEY
   ```
4. Modelos (~6 GB la primera vez):
   ```powershell
   docker compose up -d ollama qdrant
   docker compose run --rm ollama-pull
   ```
5. Imagen y verificación:
   ```powershell
   docker compose build app
   docker compose run --rm app python scripts/fase3_ingest.py
   docker compose run --rm app python scripts/fase0_healthcheck.py
   ```

Salida esperada del healthcheck:

```
[OK] Ollama accesible -> http://ollama:11434
[OK] Modelo principal -> qwen3.5:4b
[OK] Modelo fallback -> granite4.1:3b
[OK] Modelo embeddings -> qwen3-embedding:0.6b
[OK] Chat -> 'OK' en 3.2s
[OK] Tool calling -> [{'name': 'get_weather', 'args': {'city': 'Lima'}, ...}]
[OK] LangSmith -> proyecto 'ai-agentic'
[OK] Qdrant -> coleccion 'shop_policies'
```

### Opción B: entorno Python local

Ollama y Qdrant en Docker; scripts y tests con Python 3.11 local (útil para depurar desde el IDE).

```powershell
$py = "<ruta-al-entorno>\Scripts\python.exe"
& $py -m pip install -r requirements.txt -e .
& $py scripts/fase0_healthcheck.py
& $py -m pytest -q
```

`OLLAMA_BASE_URL=http://localhost:11434` en `.env`. Dentro de Docker Compose se sobrescribe con `http://ollama:11434`
(y `QDRANT_URL` con `http://qdrant:6333`).

## Modelos

Hardware de referencia: Intel i5-1135G7, 16 GB de RAM, GPU integrada Iris Xe (Ollama no la usa). Inferencia en CPU,
modelos de 3B-4B cuantizados.

| Modelo | Tamaño | Variable | Uso |
|---|---|---|---|
| `qwen3.5:4b` | 3.4 GB | `OLLAMA_MODEL` | Agentes, router y juez; tools y modo thinking |
| `granite4.1:3b` | 2.1 GB | `OLLAMA_FALLBACK_MODEL` | Fallback ante fallas del principal (Fase 6) |
| `qwen3-embedding:0.6b` | 0.6 GB | `OLLAMA_EMBED_MODEL` | Embeddings multilingües del RAG |
| Modelos de 8B-9B | 5 GB o más | - | No recomendados con 8 GB compartidos con Docker |

| Parámetro | Valor | Motivo |
|---|---|---|
| `OLLAMA_REASONING` | `false` | En CPU el modo thinking agrega segundos por llamada; se activa solo para comparar calidad y latencia |
| `OLLAMA_TEMPERATURE` | `0` | Reproducibilidad de evaluaciones |
| `OLLAMA_KEEP_ALIVE` | `30m` | Evita recargar modelos entre turnos |
| `LLM_TIMEOUT_S` | `240` | Corta llamadas colgadas sin afectar llamadas lentas (p95 por llamada ~60 s) |

Limitación de los modelos de 3B-4B: con esquemas JSON complejos, descripciones largas o muchas tools omiten
parámetros o eligen la tool equivocada. Por eso las tools son simples y las reglas de negocio no dependen del modelo.

GPU NVIDIA (opcional), en el servicio `ollama`:

```yaml
deploy:
  resources:
    reservations:
      devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
```

## Trazas

Con `LANGSMITH_TRACING=true`, cada llamada de componentes LangChain aparece en el proyecto `ai-agentic` de LangSmith
con entrada, salida, tokens y latencia.
