"""Start local servers, verify health/inference, and stop only those processes."""
import json,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]

def verify_server(kind):
    port=free_port()
    command=([sys.executable,'-m','uvicorn','api:app','--host','127.0.0.1','--port',str(port)] if kind=='api' else [sys.executable,'-m','streamlit','run','app.py','--server.address=127.0.0.1','--server.port='+str(port),'--server.headless=true'])
    health=f'http://127.0.0.1:{port}'+('/health' if kind=='api' else '/_stcore/health')
    flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
    with tempfile.TemporaryFile() as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,creationflags=flags,env={**os.environ,'MPLBACKEND':'Agg'})
        try:
            deadline=time.monotonic()+40
            while True:
                if process.poll() is not None:
                    log.seek(0);raise RuntimeError(log.read().decode(errors='replace')[-2000:])
                try:
                    with urlopen(health,timeout=2) as response:
                        body=response.read().decode()
                        assert response.status==200
                    break
                except Exception:
                    if time.monotonic()>deadline: raise
                    time.sleep(.25)
            if kind=='api':
                from src.dataset import original_split
                from src.prediction import predict
                records=original_split()[1].head(2).to_dict('records')
                request=Request(f'http://127.0.0.1:{port}/predict',data=json.dumps({'records':records}).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=10) as response:
                    prediction=json.load(response)
                assert prediction['predictions']==predict(records).to_dict('records')
                return {'health':'passed','http_inference_parity':'passed','model_id':prediction['model_id']}
            with urlopen(f'http://127.0.0.1:{port}',timeout=5) as response:
                assert response.status==200 and b'<html' in response.read().lower()
            return {'health':'passed','http_frontend':'passed'}
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)

def verify_deployment():
    return {'local_api':verify_server('api'),'local_streamlit':verify_server('streamlit'),'hosted_deployment':'Not created or independently verified.','scope':'Loopback-only local processes; stopped after checks.'}

if __name__=='__main__': print(json.dumps(verify_deployment(),indent=2))
