// Point the top-left org logo/brand at the GitHub organization.
(function () {
  var ORG_URL = "https://github.com/Cloud2BR-TEC";
  var SITE_BASE_PATH = "/ai-academy-103-ml";
  var HOME_URL = "/ai-academy-103-ml/";

  function getRelativePagePath(pathname) {
    var currentPath = pathname || "/";
    if (!currentPath.startsWith(SITE_BASE_PATH)) {
      return "/";
    }

    var pathWithoutBase = currentPath.slice(SITE_BASE_PATH.length) || "/";
    if (!pathWithoutBase.startsWith("/")) {
      pathWithoutBase = "/" + pathWithoutBase;
    }

    if (pathWithoutBase === "/es" || pathWithoutBase === "/es/") {
      return "/";
    }

    if (pathWithoutBase.startsWith("/es/")) {
      return pathWithoutBase.slice(3);
    }

    return pathWithoutBase;
  }

  function buildLanguageUrl(languageCode) {
    var relativePath = getRelativePagePath(window.location.pathname);
    var languageBase = languageCode === "es" ? SITE_BASE_PATH + "/es" : SITE_BASE_PATH;
    var targetPath = relativePath === "/" ? languageBase + "/" : languageBase + relativePath;
    return targetPath + window.location.search + window.location.hash;
  }

  function syncLanguageSelectorLinks() {
    document
      .querySelectorAll('a.md-select__link[hreflang], link[rel="alternate"][hreflang]')
      .forEach(function (link) {
        var languageCode = link.getAttribute("hreflang");
        if (languageCode === "en" || languageCode === "es") {
          link.setAttribute("href", buildLanguageUrl(languageCode));
        }
      });
  }

  function pointLogoToOrg() {
    document
      .querySelectorAll('a.md-logo, a[data-md-component="logo"], a.md-header__button.md-logo')
      .forEach(function (logo) {
        logo.setAttribute("href", ORG_URL);
        logo.setAttribute("target", "_blank");
        logo.setAttribute("rel", "noopener");
      });
  }
  function pointTitleToHome() {
    document.querySelectorAll(".md-header__title .md-ellipsis").forEach(function (title) {
      title.style.cursor = "pointer";
      title.addEventListener("click", function () {
        window.location.href = HOME_URL;
      });
    });
  }
  function init() {
    pointLogoToOrg();
    pointTitleToHome();
    syncLanguageSelectorLinks();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
