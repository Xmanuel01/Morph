# Frontend Developer Stack (v1.4)

Enkai v1.4 adds first-class frontend scaffolding and typed SDK generation.

## Commands

```text
enkai new backend <target_dir> [--api-version <v>] [--force]
enkai new frontend-chat <target_dir> [--api-version <v>] [--backend-url <url>] [--force]
enkai new fullstack-chat <target_dir> [--api-version <v>] [--backend-url <url>] [--force]
enkai sdk generate <output_file> [--api-version <v>]
```

## API Contract

Generated frontend projects and SDKs pin to a versioned backend contract:

- Base route prefix: `/api/<api_version>`
- Required header: `x-enkai-api-version: <api_version>`
- Auth header (optional): `Authorization: Bearer <token>`

Expected backend routes:

- `GET /api/<api_version>/health`
- `POST /api/<api_version>/chat`
- `GET /api/<api_version>/chat/stream`

## Scaffold Outputs

- `backend`:
  - Enkai serving project with routed HTTP middleware baseline and conversation state persistence (`conversation_state.json`).
- `frontend-chat`:
  - React + TypeScript app with streaming chat UI and typed SDK.
- `fullstack-chat`:
  - Combined `backend/` + `frontend/` layout with shared API contract defaults.

## Contract Testing

The Enkai CLI test suite validates:

- SDK generation includes version-pin header behavior.
- Backend scaffold route paths match expected versioned endpoints.
- Scaffolding commands emit required project files and env contract defaults.
- Fullstack contract flow boots generated backend + frontend artifacts and validates streaming responses.
- Conversation ID continuity across stream/chat calls remains stable under API version mismatch checks.
