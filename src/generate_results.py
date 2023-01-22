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
      const domElement = document.getElementById('brax-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""