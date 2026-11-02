(function () {
  var ACCESS_KEY = 'compMatSciTextbookAccess';
  var ACCESS_VALUE = 'verified-2026-08';
  var allowed = false;

  try {
    allowed = sessionStorage.getItem(ACCESS_KEY) === ACCESS_VALUE;
  } catch (e) {
    allowed = false;
  }

  if (!allowed) {
    document.documentElement.style.display = 'none';
    window.location.replace('../education.html#compmatsci-textbook');
  }
})();
