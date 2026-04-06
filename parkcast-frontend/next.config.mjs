/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://34.133.160.231:8000/:path*',
      },
    ];
  },
};

export default nextConfig;