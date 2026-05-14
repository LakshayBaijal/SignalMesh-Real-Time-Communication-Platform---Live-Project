import ws from 'k6/ws';
import { check } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'https://signal-mesh.onrender.com';
const HOLD_SECONDS = Number(__ENV.HOLD_SECONDS || 15);
const CONNECTED_USERS = Number(__ENV.CONNECTED_USERS || 100);

const wsConnectTime = new Trend('ws_connect_time_ms', true);
const wsConnections = new Counter('ws_connections');
const wsConnectionChecks = new Rate('ws_connection_checks');

export const options = {
  scenarios: {
    ws_stress: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '30s', target: 25 },
        { duration: '45s', target: 50 },
        { duration: '60s', target: CONNECTED_USERS },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
};

function toWsUrl(url) {
  if (url.startsWith('https://')) {
    return `wss://${url.slice('https://'.length)}`;
  }
  if (url.startsWith('http://')) {
    return `ws://${url.slice('http://'.length)}`;
  }
  return url;
}

export default function () {
  const connectStart = Date.now();
  const url = `${toWsUrl(BASE_URL)}/ws`;

  const res = ws.connect(url, {}, function (socket) {
    socket.on('open', function () {
      wsConnectTime.add(Date.now() - connectStart);
      wsConnections.add(1);
      socket.setTimeout(function () {
        socket.close();
      }, HOLD_SECONDS * 1000);
    });

    socket.on('error', function () {
      wsConnectionChecks.add(false);
    });
  });

  check(res, { 'websocket connected': (r) => r && r.status === 101 });
  wsConnectionChecks.add(res && res.status === 101);
}