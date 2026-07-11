# Dental AI

## Status

AI inference microservice scaffolded for the first detection module: **tooth
detection + FDI numbering**. No model is trained yet — that's blocked on
annotated data (see below).

```
services/inference/       Python inference microservice (isolated, called only by Convex)
    app.py                 FastAPI app — /v1/infer endpoint
    inference.py            ToothDetectionPipeline
    preprocessing/           image/DICOM loading + normalization
    postprocessing/          FDI numbering scheme (11-48)
    report_engine/           response schema shared with Convex
    training/                train_tooth_detection.py
    models/tooth_detection/  checkpoints land here after training
    tests/                   unit tests (passing)

datasets/tooth_detection/  🚩 needs annotated data — see datasets/tooth_detection/README.md
```

Architecture follows `Documentation/requirements.md`: Convex is the backend
for everything except GPU inference; this Python service never talks to
users directly, only to Convex via the job queue.

## 🚩 Blocked on annotated data

**`datasets/tooth_detection/README.md`** has the full spec: image format,
annotation format (YOLO bounding boxes labeled with FDI tooth numbers),
recommended annotation tools, and folder layout. Short version — I need:

1. De-identified panoramic X-rays (150+ minimum, 300-500+ preferred) in
   `datasets/tooth_detection/raw_images/`.
2. Per-tooth bounding box annotations in `datasets/tooth_detection/annotations/`
   (or raw images + your annotation tool's export format — I'll write a
   converter).

Nothing downstream (caries detection, missing-tooth detection, clinical
decision engine, quotation engine) can be trained or validated until this
first model exists, since every later phase in the roadmap depends on
tooth localization.

## Web application (apps/web)

Next.js 16 + TypeScript + Tailwind + shadcn/ui app scaffolded, with the
mandated stack installed (React Hook Form, Zod, TanStack Table, Recharts,
Framer Motion, Convex). Feature-folder structure created for every module in
the roadmap (`src/features/{authentication,dashboard,cases,reports,
quotations,tenants,clinics,pricing,ai,analytics,settings}`), each with
`components/hooks/types/schemas/actions/utils/tests`.

Working end-to-end right now: a dashboard page that queries Convex's
`cases.listCases` and renders a table (empty until real data exists).

Convex backend (`apps/web/convex/`) is written per `Documentation/requirements.md`:
`schema.ts` (tenants, clinics, users, patients, cases, images, aiJobs,
reports, pricingRules, quotations, auditLogs — every table tenant-scoped),
plus function files for tenants, users, cases, aiJobs, reports, pricing,
quotations, auditLogs, and a `lib/tenant.ts` helper that enforces tenant
isolation and RBAC on every query/mutation.

### Auth status

`npx convex dev` has been run (you did this) and the real Convex Auth
initializer (`npx @convex-dev/auth`) has been run too — this generated
`convex/http.ts` and set the `SITE_URL`/`JWT_PRIVATE_KEY`/`JWKS` env vars on
the dev deployment that Convex Auth needs to issue tokens. `convex/schema.ts`
now spreads `authTables` and extends its `users` table with `tenantId`/`role`
for RBAC. The app renders and a **Password provider** is wired up in
`convex/auth.ts` so local sign-up/sign-in works today without external
credentials.

The root layout needed `ConvexAuthNextjsServerProvider` (an async Server
Component wrapper that fetches server-side auth state) around the client
`Providers` — without it, `ConvexAuthNextjsProvider`'s internal `useAuth()`
hook has no context to read and throws. That's now in
`src/app/layout.tsx`.

### 🚩 Still needs OAuth/email credentials

Password auth works, but requirements.md calls for Email OTP + Google +
Microsoft. Those need real credentials before they can replace/join Password:
`AUTH_GOOGLE_ID`/`SECRET`, `AUTH_MICROSOFT_ENTRA_ID_ID`/`SECRET`,
`AUTH_RESEND_KEY`. See `apps/web/.env.local.example`.

### 🚩 Not yet built: sign-in UI

`convex/auth.ts` and the provider wiring are functional, but there's no
`/sign-in` page or form yet — `useAuthActions()` (`signIn`/`signOut`) isn't
called from any component yet.

### Running the web app

```
cd apps/web
npm run dev
```

## Not yet started

Auth UI (sign-in pages), RBAC enforcement in the UI layer, clinic/patient
management, upload flow, pricing/quotation UI, PDF generation, and the
remaining detection modules (caries, crowns, missing teeth, periapical
lesions, etc.) — each new detection module will need its own annotated
dataset, flagged the same way as tooth detection above when we get there.

## Running the inference service locally

```
cd services/inference
pip install -r requirements.txt
uvicorn app:app --reload
```

`/v1/infer` will raise a clear error until a trained checkpoint exists at
`models/tooth_detection/weights/best.pt`.
