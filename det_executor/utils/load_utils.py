import wget

SCRIPT_16 = '1L8mPcUvabUscEk6Nr8ck5EFgopgPAMDW'
SCRIPT_16_TINY = '18zJyljtolPENDI_kFw3FlRFnQTnaLuDF'

def load_script_model(link, out):
	path = f'https://drive.google.com/uc?export=download&id={link}&confirm=t'
	wget.download(path, out, bar=None)