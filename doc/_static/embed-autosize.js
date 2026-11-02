/*
  Fit the embedded-output iframes (_static/embed/*.html) to their content.

  The pages embed two different kinds of document, and a single fixed `height`
  attribute cannot serve both:

    * flow documents (molecule_to_crystal_simulator, MolOD_*, CrystalOD_*,
      Multiplet_*) have a natural height that depends on how wide the frame is
      -- their grids stack below ~820px -- so any fixed height clips them at
      some window size.  Measured at a 784px-wide frame, the landing-page
      embeds wanted 849 / 1101 / 704px against declared heights of 640 / 660 /
      660: everything below the fold was simply unreachable.

    * the 3D viewers (SALC_*, BZ_*) are laid out in viewport units
      (`height:100vh` / `95vh`), which inside an iframe means "the height of
      the frame".  They already fill whatever they are given, and their
      scrollHeight only ever mirrors the frame, so they must be left alone.

  Both are handled by measuring the content once with the frame collapsed: a
  flow document keeps its full scrollHeight, a viewport-sized one collapses
  with the frame and is recognised by that.  The measurement is done and undone
  within one task, so the browser never paints the collapsed state.

  Same-origin only -- everything under _static/embed/ is served from this site,
  and anything cross-origin is skipped by the try/catch.
*/

(function () {
  "use strict";

  // the content is measured at two very different frame heights; a document
  // whose height is its own gives the same answer twice, one laid out in
  // viewport units follows the frame
  var PROBE_SMALL = 60;
  var PROBE_LARGE = 2000;
  var TOLERANCE = 2;

  function embedFrames() {
    return Array.prototype.filter.call(
      document.querySelectorAll("iframe"),
      function (frame) {
        return frame.src.indexOf("/_static/embed/") !== -1;
      }
    );
  }

  function contentHeight(doc) {
    return Math.max(
      doc.documentElement ? doc.documentElement.scrollHeight : 0,
      doc.body ? doc.body.scrollHeight : 0
    );
  }

  function fit(frame) {
    var doc;
    try {
      doc = frame.contentDocument;
    } catch (e) {
      return; // cross-origin: nothing we can do, keep the declared height
    }
    if (!doc || !doc.body) {
      return;
    }

    // the height declared in the .md file is the floor for anything that
    // sizes itself against the frame
    if (!frame.dataset.declaredHeight) {
      frame.dataset.declaredHeight =
        frame.getAttribute("height") || String(frame.getBoundingClientRect().height);
    }
    var declared = parseFloat(frame.dataset.declaredHeight) || 0;

    frame.style.height = PROBE_SMALL + "px";
    var small = contentHeight(doc);
    frame.style.height = PROBE_LARGE + "px";
    var large = contentHeight(doc);

    if (Math.abs(large - small) <= TOLERANCE) {
      // height independent of the frame: fit it exactly, which both reveals
      // what a too-short frame was hiding and reclaims a too-tall one
      frame.style.height = Math.ceil(small) + "px";
    } else {
      // the document stretches with the frame (95vh/100vh viewers, possibly
      // with flow content above them): never shrink it below what the page
      // asked for, but still grow if the flow part alone needs more
      frame.style.height = Math.ceil(Math.max(declared, small)) + "px";
    }
  }

  function fitAll() {
    embedFrames().forEach(fit);
  }

  function watch(frame) {
    // `loading="lazy"` means load fires when the frame scrolls into view
    frame.addEventListener("load", function () {
      fit(frame);
      // the content can settle after load (fonts, canvas sizing); re-fit once
      // the browser has laid it out
      requestAnimationFrame(function () {
        fit(frame);
      });
    });
    if (frame.contentDocument && frame.contentDocument.readyState === "complete") {
      fit(frame);
    }
  }

  function start() {
    var frames = embedFrames();
    frames.forEach(watch);

    // the natural height depends on the frame width (the embeds restack at
    // ~820px), so re-fit whenever the column is resized -- this also covers
    // the right-hand table of contents being toggled
    if (window.ResizeObserver && frames.length) {
      var pending = null;
      var observer = new ResizeObserver(function () {
        if (pending) {
          return;
        }
        pending = requestAnimationFrame(function () {
          pending = null;
          fitAll();
        });
      });
      frames.forEach(function (frame) {
        if (frame.parentElement) {
          observer.observe(frame.parentElement);
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
