/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === "production";

const nextConfig = {
  output: "export",
  basePath: isProd ? "/ticker-talk" : "",
  assetPrefix: isProd ? "/ticker-talk" : "",
};

module.exports = nextConfig;
