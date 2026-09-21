import type { NextConfig } from "next";

// Static export: `npm run build` writes web/out, which the FastAPI server
// serves at / so the whole product is one process.  During `next dev` the
// dashboard talks to the API on :8000 via NEXT_PUBLIC_API_URL.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
