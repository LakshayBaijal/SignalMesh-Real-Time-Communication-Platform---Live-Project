import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'https://signal-mesh.onrender.com';
const PASSWORD = __ENV.BENCH_PASSWORD || 'BenchPass123!';
const WS_HOLD_SECONDS = Number(__ENV.WS_HOLD_SECONDS || 15);
const MESSAGE_INTERVAL_SECONDS = Number(__ENV.MESSAGE_INTERVAL_SECONDS || 3);

const wsConnectTime = new Trend('ws_connect_time_ms', true);
const wsRoundTrip = new Trend('ws_roundtrip_ms', true);
const wsMessagesSent = new Counter('ws_messages_sent');
const wsMessagesReceived = new Counter('ws_messages_received');
const wsErrors = new Counter('ws_errors');
const wsCheckRate = new Rate('ws_checks');

export const options = {
  scenarios: {
    ws_auth_chat: {
      executor: 'constant-vus',
      vus: 1,
      duration: '20s',
      startTime: '2s',
    },
  },
};

function joinUrl(base, path) {
  const left = base.endsWith('/') ? base.slice(0, -1) : base;
  const right = path.startsWith('/') ? path : `/${path}`;
  return `${left}${right}`;
}

function toWsUrl(url) {
  if (url.startsWith('https://')) {
    return `wss://${url.slice('https://'.length)}`;
  }
  if (url.startsWith('http://')) {
    return `ws://${url.slice('http://'.length)}`;
  }
  return url;
}

export function setup() {
  const headers = { headers: { 'Content-Type': 'application/json' } };
  const username = __ENV.BENCH_USERNAME || 'bench_login_user';
  const signupUrl = joinUrl(BASE_URL, '/signup');
  const loginUrl = joinUrl(BASE_URL, '/login');
  const payload = JSON.stringify({ username, password: PASSWORD });

  const signupResponse = http.post(signupUrl, payload, headers);
  check(signupResponse, {
    'setup signup accepted (200 or 400)': (r) => r.status === 200 || r.status === 400,
  });

  const loginResponse = http.post(loginUrl, payload, headers);
  const loginOk = loginResponse.status === 200;
  let token = null;

  check(loginResponse, {
    'setup login status is 200': () => loginOk,
    'setup login has token': (r) => {
      try {
        token = r.json('token');
        return Boolean(token);
      } catch (error) {
        return false;
      }
    },
  });

  return {
    users: loginOk && token ? [{ username, token }] : [],
    createdAt: Date.now(),
  };
}

export default function (data) {
  if (!data || !data.users || data.users.length === 0) {
    wsErrors.add(1);
    sleep(1);
    return;
  }

  const user = data.users[(__VU - 1) % data.users.length];
  const wsBase = toWsUrl(BASE_URL);
  const wsUrl = joinUrl(wsBase, `/ws/${user.token}`);

  const connectStart = Date.now();

  const response = ws.connect(wsUrl, {}, function (socket) {
    socket.on('open', () => {
      wsConnectTime.add(Date.now() - connectStart);

      socket.setInterval(() => {
        const now = Date.now();
        const payload = {
          type: 'message',
          name: user.username,
          text: `bench-msg-${__VU}-${now}`,
          _sentAt: now,
        };
        socket.send(JSON.stringify(payload));
        wsMessagesSent.add(1);
      }, MESSAGE_INTERVAL_SECONDS * 1000);

      socket.setTimeout(() => {
        socket.close();
      }, WS_HOLD_SECONDS * 1000);
    });

    socket.on('message', (message) => {
      wsMessagesReceived.add(1);

      try {
        const parsed = JSON.parse(message);
        if (parsed && parsed.type === 'message' && parsed.name === user.username) {
          const text = String(parsed.text || '');
          const marker = text.split('-').pop();
          const sentAt = Number(marker);
          if (!Number.isNaN(sentAt) && sentAt > 0) {
            wsRoundTrip.add(Date.now() - sentAt);
          }
        }
      } catch (error) {
        wsErrors.add(1);
      }
    });

    socket.on('error', () => {
      wsErrors.add(1);
    });
  });

  const ok = check(response, {
    'websocket upgrade status 101': (r) => r && r.status === 101,
  });
  wsCheckRate.add(ok);

  sleep(0.5);
}
