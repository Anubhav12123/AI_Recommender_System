from cachetools import TTLCache

# Simple in-process cache (could swap for Redis)
search_cache = TTLCache(maxsize=1024, ttl=60)
