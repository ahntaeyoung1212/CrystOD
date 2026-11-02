/*
  Desktop show/hide for the right-hand "Contents" sidebar.

  Counterpart of the handler sphinx-book-theme installs for the left
  navigation sidebar: from 1200px up -- the width at which the secondary
  sidebar stops being a modal drawer and is laid out inline -- clicking
  .secondary-toggle toggles .pst-sidebar-hidden on #pst-secondary-sidebar
  instead of opening pydata-sphinx-theme's mobile modal.  Below that width the
  theme's own modal behaviour is left untouched.

  The listener sits on `document` in the capture phase so that it runs before
  the theme's listeners on the button itself, whatever order the scripts
  happen to load in; stopPropagation() then keeps pydata from moving the
  sidebar contents into <dialog id="pst-secondary-sidebar-modal">.

  See _static/sidebar-toggle.css for the matching rules.
*/

(function () {
  "use strict";

  // the breakpoint below which pydata-sphinx-theme makes the secondary
  // sidebar a fixed drawer (@media (max-width: 1199.98px))
  var INLINE_SIDEBAR = "(min-width: 1200px)";

  document.addEventListener(
    "click",
    function (event) {
      var target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      var button = target.closest(".secondary-toggle");
      if (!button || !window.matchMedia(INLINE_SIDEBAR).matches) {
        return;
      }
      var sidebar = document.querySelector("#pst-secondary-sidebar");
      if (!sidebar) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      var hidden = sidebar.classList.toggle("pst-sidebar-hidden");
      // the article container is an earlier sibling of the sidebar, so CSS
      // cannot reach it from here; carry the state on <body> instead
      document.body.classList.toggle("crystod-toc-hidden", hidden);
      button.setAttribute("aria-expanded", hidden ? "false" : "true");
    },
    true
  );
})();
