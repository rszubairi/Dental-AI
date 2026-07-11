Yes. In fact, since the project is intended to be built primarily using **AI coding agents**, I would make the specification much more prescriptive so that every agent follows the same architecture and doesn't invent its own technology stack.

I would explicitly instruct the AI agents that **Next.js is the only frontend framework** and **Convex is the only backend platform**. The document should discourage introducing unnecessary backend frameworks (such as FastAPI, Express, NestJS, or Django) unless specifically required for GPU inference.

Below is an enhanced version of the technology and implementation directives that can become the foundation of your specification.

---

# AI Coding Agent Development Directives

## Purpose

This project shall be developed primarily using AI Coding Agents (Claude Code, GPT-5 Codex, Cursor, Windsurf, Cline, Roo Code, etc.). Every AI agent must follow this specification exactly.

The objective is to produce a maintainable, production-grade enterprise application using a single, consistent architecture.

AI agents shall not substitute alternative frameworks unless explicitly instructed.

---

# Mandatory Technology Stack

## Frontend

The frontend MUST be developed using:

* Next.js 15+
* React 19+
* TypeScript
* Tailwind CSS
* shadcn/ui
* React Hook Form
* Zod
* TanStack Table
* Recharts
* Framer Motion

The application must use the Next.js App Router.

No Pages Router shall be used.

Server Components shall be used whenever possible.

Client Components shall only be used when necessary.

---

## Backend

The backend MUST be developed entirely using Convex.

Convex shall be responsible for:

* Authentication
* Database
* File Metadata
* Case Management
* Multi-Tenant Management
* Audit Logs
* User Management
* Pricing Engine
* Quotation Storage
* AI Job Queue
* Notifications
* Role Based Access Control
* API Layer
* Realtime Updates

No separate backend framework shall be introduced unless specifically required for GPU inference.

---

## AI Inference Service

The AI inference pipeline shall remain an isolated Python microservice.

Responsibilities include:

* Model Loading
* GPU Inference
* Image Processing
* Model Selection
* Report Generation
* Confidence Calculation

The inference service shall never communicate directly with users.

Instead:

Next.js

↓

Convex

↓

Inference Queue

↓

Python AI Service

↓

Convex

↓

Next.js

---

## Storage

Use Convex File Storage for

* X-Ray Uploads
* Reports
* PDF Quotations
* Temporary Images

Use Blob Storage when datasets exceed Convex limitations.

---

# Repository Structure

```text
apps/
    web/

convex/
    auth.ts
    users.ts
    tenants.ts
    cases.ts
    reports.ts
    pricing.ts
    quotations.ts
    aiJobs.ts
    auditLogs.ts
    schema.ts

packages/
    ui/
    shared/
    ai/
    pricing/
    clinical/

services/
    inference/

        app.py
        inference.py
        models/
        preprocessing/
        postprocessing/
        report_engine/
        requirements.txt

docs/

infra/
```

---

# AI Agent Rules

Every AI coding agent MUST follow these rules.

## Rule 1

Never introduce another backend framework.

Convex is the backend.

---

## Rule 2

Never duplicate business logic.

Business rules belong inside Convex Functions.

---

## Rule 3

Never hardcode values.

Everything configurable must live inside Convex.

Examples:

Treatment pricing

Currencies

Countries

Confidence thresholds

Clinic configuration

Role permissions

---

## Rule 4

Never call the AI inference service directly from React.

Always go through Convex.

```
React

↓

Convex Mutation

↓

Inference Queue

↓

Python AI

↓

Convex

↓

React Subscription
```

---

## Rule 5

Every feature must be modular.

Example

```
/features

    authentication

    dashboard

    cases

    reports

    quotations

    tenants

    clinics

    pricing

    ai

    analytics

    settings
```

---

## Rule 6

Every feature shall contain

```text
components/

hooks/

types/

schemas/

actions/

utils/

tests/
```

---

## Rule 7

Never write SQL.

Convex is the source of truth.

---

## Rule 8

Never access Convex directly from UI Components.

Always use

Queries

Mutations

Actions

---

## Rule 9

All validation shall use Zod.

No manual validation.

---

## Rule 10

Every API response shall be strongly typed.

No use of `any`.

---

# Multi-Tenant Architecture

The application shall support unlimited organizations.

```
Platform

├── Tenant

│      ├── Clinics

│      ├── Users

│      ├── Patients

│      ├── Cases

│      ├── AI Reports

│      ├── Quotations

│      ├── Billing

│      └── Settings
```

Every Convex document shall include

```typescript
tenantId
```

Every query must automatically filter by

```typescript
tenantId
```

Cross-tenant access is prohibited.

---

# Authentication

Authentication shall use Convex Auth.

Supported providers

* Email OTP
* Google
* Microsoft

Roles

```
Super Admin

Tenant Admin

Clinic Manager

Dentist

Call Centre Agent

Receptionist

Auditor

Viewer
```

RBAC shall be enforced in Convex.

Never inside React.

---

# AI Processing Workflow

```
Upload X-ray

↓

Convex Storage

↓

Create Case

↓

Generate Case ID

↓

Queue AI Job

↓

Python Inference Service

↓

Run Detection Models

↓

Clinical Rule Engine

↓

Treatment Recommendation

↓

Cost Estimation

↓

Quotation Generator

↓

Save Report

↓

Notify User

↓

Display Results
```

---

# Development Phases

## Phase 1

Foundation

* Next.js
* Convex
* Authentication
* RBAC
* Multi-Tenant
* Design System

---

## Phase 2

Case Management

* Patient
* Clinic
* Upload
* File Storage
* Dashboard

---

## Phase 3

AI Infrastructure

* Queue
* Python Service
* GPU
* Model Registry
* Monitoring

---

## Phase 4

Detection Models

* Tooth Detection
* FDI Numbering
* Caries
* Crowns
* Missing Teeth
* Periapical Lesions

---

## Phase 5

Clinical Decision Engine

* Treatment Recommendation
* Risk Scoring
* Confidence Levels

---

## Phase 6

Quotation Engine

* Pricing Rules
* Currency Support
* Discounts
* Insurance
* PDF Generation

---

## Phase 7

Reporting

* AI Report Viewer
* Timeline
* Case History
* Export
* Audit Logs

---

## Phase 8

Production Readiness

* Security
* Monitoring
* Performance
* Load Testing
* Deployment
* CI/CD

## AI Coding Agent Success Criteria

Every pull request generated by an AI coding agent must satisfy the following before it can be merged:

* Follow the prescribed Next.js + Convex architecture without introducing additional backend frameworks.
* Use TypeScript with strict typing enabled; no `any` types.
* Pass ESLint, Prettier, and all automated tests.
* Include unit tests for business logic and integration tests for Convex functions.
* Maintain tenant isolation in every query and mutation.
* Validate all inputs using Zod schemas.
* Include documentation updates where functionality changes.
* Ensure reusable components are added to the shared UI library instead of duplicating code.
* Avoid hardcoded configuration values; all configurable settings must be stored in Convex or environment variables.
* Follow SOLID principles, feature-based folder organization, and consistent naming conventions.

This approach gives AI coding agents a clear architectural contract, significantly reducing inconsistent implementations while making the platform easier to maintain and extend as new AI models, treatment workflows, and tenant-specific features are added.
