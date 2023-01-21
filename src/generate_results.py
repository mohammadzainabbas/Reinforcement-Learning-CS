_HTML = """
<html>
  <head>
    <title>brax visualizer</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }
      #brax-viewer {
        margin: 0;
        padding: 0;
        height: <!-- viewer height goes here -->;
      }
    </style>
  </head>
  <body>
    <script type="application/javascript">
    var system = <!-- system json goes here -->;
    </script>
    <div id="brax-viewer"></div>
    <script type="module">
      import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.1.0/js/viewer.js';
      const domElement = document.getElementById('brax-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""