// Egyszerű cache-first service worker PWA-hoz

const CACHE_NAME = "insurance-talk-cache-v1";
const URLS_TO_CACHE = [
  "/",
  "/?source=pwa",
  "/manifest.json"
  // Ide tehetsz még CSS/JS/logo URL-eket, ha statikusan elérhetők
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(URLS_TO_CACHE);
    })
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== CACHE_NAME)
          .map(k => caches.delete(k))
      )
    )
  );
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    caches.match(event.request).then(response => {
      // Cache-first, ha nincs cache, megy hálózatra
      return response || fetch(event.request);
    })
  );
});
