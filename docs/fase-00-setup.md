# Fase 0 — Setup

## Objetivo

Tener Ollama, el modelo, el tool calling y LangSmith funcionando antes de escribir logica agentica.

## Componentes

| Servicio | Rol |
|---|---|
| `ollama` | Servidor de modelos en `:11434`. Los modelos persisten en el volumen `ollama_data`. |
| `ollama-pull` | Job de una ejecucion: descarga `OLLAMA_MODEL`, `OLLAMA_FALLBACK_MODEL` y `OLLAMA_EMBED_MODEL`. |
| `qdrant` | Vector DB en `:6333` (Fase 3). Datos en el volumen `qdrant_data`. |
| `app` | Contenedor Python donde corren scripts y tests. Monta el proyecto en `/app`, asi que los cambios de codigo no requieren rebuild. |
| LangSmith | SaaS (smith.langchain.com). La version self-hosted requiere licencia enterprise, por eso se usa la nube (plan gratuito). |

## Pasos

1. Instalar **Docker Desktop** (Windows, backend WSL2) y verificar con `docker --version`.
   Limitar la memoria de WSL2 creando `C:\Users\<usuario>\.wslconfig` (luego `wsl --shutdown`):
   ```ini
   [wsl2]
   memory=8GB
   processors=6
   ```
2. Crear cuenta en https://smith.langchain.com → Settings → API Keys → crear key.
3. Configurar variables:
   ```powershell
   Copy-Item .env.example .env
   # editar LANGSMITH_API_KEY
   ```
4. Levantar Ollama y descargar los modelos (~5.5 GB la primera vez):
   ```powershell
   docker compose up -d ollama
   docker compose run --rm ollama-pull
   ```
5. Build de la app y healthcheck:
   ```powershell
   docker compose build app
   docker compose run --rm app python scripts/fase0_healthcheck.py
   ```

Salida esperada:
```
[OK] Ollama accesible -> http://ollama:11434
[OK] Modelo principal -> qwen3.5:4b
[OK] Modelo fallback -> granite4.1:3b
[OK] Chat -> 'OK' en 3.2s
[OK] Tool calling -> [{'name': 'get_weather', 'args': {'city': 'Lima'}, ...}]
[OK] LangSmith -> proyecto 'ai-agentic'
```

### Opcion B: entorno local `env_ml`

Ollama en Docker; el codigo corre con tu Python local (mas rapido para iterar y depurar en VS Code).

```powershell
$py = "C:\Users\CristhianSanchez(Con\Desktop\ism\env_ml\Scripts\python.exe"
& $py -m pip install -r requirements.txt -e .
& $py scripts/fase0_healthcheck.py
& $py -m pytest -q
```

En `.env` debe quedar `OLLAMA_BASE_URL=http://localhost:11434`. Dentro de Compose se sobrescribe a `http://ollama:11434`.

## Eleccion de modelo

Hardware objetivo: i5-1135G7, 16 GB RAM, Iris Xe (Ollama no la usa) → inferencia en CPU, rango 3B-4B en Q4.

| Modelo | Tamano | Rol |
|---|---|---|
| `qwen3.5:4b` | 3.4 GB | Principal (`OLLAMA_MODEL`). Tools + thinking. |
| `granite4.1:3b` | 2.1 GB | Fallback (`OLLAMA_FALLBACK_MODEL`). Entrenado para tool use / RAG; mas rapido. |
| 8B-9B | 5+ GB | No recomendado con 16 GB compartidos con Docker. |

**Reasoning.** `OLLAMA_REASONING=false` desactiva el modo thinking de `qwen3.5` (`ChatOllama(reasoning=False)`).
En CPU el razonamiento agrega segundos por llamada; activarlo solo para comparar calidad vs latencia.

**Limitaciones de modelos pequenos.** Con JSON Schema complejo, descripciones largas o varias tools
tienden a omitir parametros o elegir la tool equivocada. Mantener tools simples y medir estas fallas en Fase 4.

Para cambiar de modelo: editar `.env` y ejecutar `docker compose run --rm ollama-pull`.
`get_chat_model(fallback=True)` devuelve el modelo alternativo.

Con GPU NVIDIA, agregar al servicio `ollama`:
```yaml
deploy:
  resources:
    reservations:
      devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
```

## Que revisar en LangSmith

Abrir el proyecto `ai-agentic`: cada llamada del healthcheck aparece como un run con input, output, tokens y latencia.
La trazabilidad es automatica para componentes LangChain cuando `LANGSMITH_TRACING=true`.

## Checklist

- [ ] Explicar por que el modelo y la BD de trazas estan desacoplados de la app (settings por variables de entorno).
- [ ] Explicar la diferencia entre `ollama` y `ollama-pull` y el uso de `depends_on` con `condition`.
- [ ] Ubicar en LangSmith la traza del check de tool calling y leer el `tool_call` generado.
