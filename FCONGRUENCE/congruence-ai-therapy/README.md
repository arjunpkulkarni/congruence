# 🧠 Congruence — AI-Powered Therapy Intelligence

AI-powered session insights, emotion timelines, and clinical note generation for mental health professionals.

---

## 🚀 Tech Stack

- React + Vite
- TypeScript
- Tailwind CSS + shadcn-ui
- Supabase (Auth, Database, Storage, Edge Functions)

---

## ⚙️ Development Setup

```bash
# 1. Clone the repository
git clone <YOUR_GIT_URL>

# 2. Navigate into the project
cd congruence-ai-therapy

# 3. Install dependencies
npm install

# 4. Start frontend
npm run dev
```

---

## 🔐 Environment Variables

Create a `.env` file in the root:

```bash
VITE_SUPABASE_URL=https://jpraokydxooziiryacid.supabase.co
VITE_SUPABASE_ANON_KEY=your_publishable_key_here
```

---

## 🧩 Supabase Development

### 🔑 Login

```bash
npx supabase login
```

### 🔗 Link Project

```bash
npx supabase link --project-ref jpraokydxooziiryacid
```

### 🧪 Start Local Supabase

```bash
npx supabase start
```

---

## 🗄️ Database (Migrations)

```bash
# Create migration
npx supabase migration new create_core_tables

# Apply locally
npx supabase db reset

# Push to remote
npx supabase db push
```

---

## ⚡ Edge Functions

```bash
# Create function
npx supabase functions new generate-treatment-plan

# Run locally
npx supabase functions serve

# Deploy all
npx supabase functions deploy

# Deploy specific
npx supabase functions deploy generate-treatment-plan

# Public webhook (no JWT)
npx supabase functions deploy generate-treatment-plan --no-verify-jwt
```

---

## 📦 Available Scripts

- `npm run dev` → Start development server  
- `npm run build` → Build for production  
- `npm run build:dev` → Build in development mode  
- `npm run lint` → Run ESLint  
- `npm run preview` → Preview production build  

---

## 🧠 Architecture Notes

- Supabase = backend (DB, auth, functions)
- React = frontend
- Edge Functions = NextJS backend (Operations backend)
- Python (DIGITAL OCEAN URL) = api.congruenceinsights.com (AI operations)

---

## 🚨 Important

- Never expose service role keys in frontend  
- Always use Edge Functions for AI / sensitive logic  
- Use migrations for schema changes  
- Enable RLS for all clinical data  

---

## 🔥 TLDR

```bash
npm install
npx supabase login
npx supabase link --project-ref jpraokydxooziiryacid
npx supabase start
npm run dev
```
