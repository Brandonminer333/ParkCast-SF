/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: (process.env.NEXT_PUBLIC_API_BASE || 'https://parkcast-api-904706413856.us-central1.run.app') + '/:path*',
      },
    ];
  },
};
export default nextConfig;
