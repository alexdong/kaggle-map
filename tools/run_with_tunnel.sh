#!/bin/bash

# Script to run piste-mind web interface with Cloudflare tunnel

echo "🚀 Starting Piste Mind web interface with Cloudflare tunnel..."

echo "🌐 Starting authenticated Cloudflare tunnel..."
echo "📍 Your app will be available at: https://piste-mind.alexdong.com"
cloudflared tunnel --config cloudflare-tunnel.yml run
else
    echo "🌐 Starting temporary Cloudflare tunnel..."
    cloudflared tunnel --url http://localhost:8000
fi

# When cloudflared is terminated, also stop the web server
trap "echo '🛑 Stopping web server...'; kill $WEB_PID 2>/dev/null" EXIT