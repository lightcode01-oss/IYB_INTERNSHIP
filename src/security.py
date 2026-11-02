"""Bound API payloads before parsing and check the trusted model checksum."""
import hashlib
from pathlib import Path

MAX_BODY_BYTES=1024*1024

def verify_model(path,checksum_path):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected=Path(checksum_path).read_text(encoding='utf-8').strip().split()[0]
    if actual!=expected: raise RuntimeError('Model checksum mismatch; restore the trusted artifact')
    return actual

class BodyLimitMiddleware:
    def __init__(self,app,max_bytes=MAX_BODY_BYTES):
        self.app=app;self.max_bytes=max_bytes
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        from starlette.responses import JSONResponse
        messages=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect': return
            size+=len(message.get('body',b''))
            if size>self.max_bytes:
                return await JSONResponse({'detail':'Request exceeds 1 MiB'},status_code=413)(scope,receive,send)
            messages.append(message)
            if not message.get('more_body',False): break
        async def replay():
            if messages: return messages.pop(0)
            return await receive()
        await self.app(scope,replay,send)
