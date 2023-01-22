from json import loads
from typing import Union, Dict, Any
from pathlib import Path
from os.path import join

def read_json(path: Union[str, Path]) -> Dict[str, Any]:
	with open(path, "r") as f:
		return loads(f.read())

def main() -> None:
	# Read the JSON files

	results_dir = Path(__file__).resolve().parents[1]

	print(results_dir)

	# with open("results/initial_system.json", "r") as f:
	# 	initial_system = loads(f.read())

	# with open("results/sys_1K.json", "r") as f:
	# 	sys_1K = loads(f.read())

	# with open("results/sys_5M.json", "r") as f:
	# 	sys_5M = loads(f.read())

	# with open("results/sys_400M.json", "r") as f:
	# 	sys_400M = loads(f.read())

	# with open("results/final_600M.json", "r") as f:
	# 	final_600M = loads(f.read())

	# with open("results/index.html", "w") as f:
	# 	f.write(_HTML.replace("<!-- initial_system -->", str(initial_system))
	# 						.replace("<!-- sys_1K -->", str(sys_1K))
	# 						.replace("<!-- sys_5M -->", str(sys_5M))
	# 						.replace("<!-- sys_400M -->", str(sys_400M))
	# 						.replace("<!-- final_600M -->", str(final_600M)))

_HTML = """
<html>
  <head>
    <title>Grasp - Pick-and-place robotic hand (Visualisation)</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }
      #brax-viewer {
        margin: 0;
        padding: 0;
        height: 100vh;
      }
    </style>
  </head>
  <body>
    <script type="application/javascript">

    var initial_system = <!-- initial_system -->;
    var sys_1K = <!-- sys_1K -->;
    var sys_5M = <!-- sys_5M -->;
    var sys_400M = <!-- sys_400M -->;
    var final_600M = <!-- final_600M -->;
    
	</script>

	<h3>1. Initial Agent's State </h3>
    <div id="brax-viewer-initial_system"></div>
	
	<h3>2. Agent's state after 1K steps</h3>
    <div id="brax-viewer-sys_1K"></div>
	
	<h3>3. Agent's state after 5 million steps</h3>
    <div id="brax-viewer-sys_5M"></div>
	
	<h3>4. Agent's state after 400 million steps</h3>
    <div id="brax-viewer-sys_400M"></div>
	
	<h3>5. Final Agent's state (after 600 million steps)</h3>
    <div id="brax-viewer-final_600M"></div>
    
	<script type="module">

      import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.1.0/js/viewer.js';

      const domElement_initial_system = document.getElementById('brax-viewer-initial_system');
      var viewer_initial_system = new Viewer(domElement_initial_system, initial_system);

      const domElement_sys_1K = document.getElementById('brax-viewer-sys_1K');
      var viewer_sys_1K = new Viewer(domElement_sys_1K, sys_1K);

      const domElement_sys_5M = document.getElementById('brax-viewer-sys_5M');
      var viewer_sys_5M = new Viewer(domElement_sys_5M, sys_5M);

      const domElement_sys_400M = document.getElementById('brax-viewer-sys_400M');
      var viewerys_400M = new Viewer(domElement_sys_400M, sys_400M);

      const domElement_final_600M = document.getElementById('brax-viewer-final_600M');
      var vieweral_600M = new Viewer(domElement_final_600M, final_600M);

    </script>
  </body>
</html>
"""