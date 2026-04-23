/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://parkcast-api-904706413856.us-central1.run.app/:path*',
      },
    ];
  },
};
export default nextConfig;
