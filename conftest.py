"""Hace que `import src...` funcione desde los tests.

Sin esto, `pytest tests/` falla con "No module named 'src'": pytest no anade
la raiz del proyecto al path. Con `python -m pytest` si funciona, y esa
diferencia despista mucho.

Con este fichero en la raiz, los tests corren igual desde la terminal, desde
el editor y desde integracion continua.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
