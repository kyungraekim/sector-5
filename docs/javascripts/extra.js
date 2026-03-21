// Smooth scroll for anchor links
// The docs have manually-written TOC lists at the top of many pages;
// smooth scrolling makes clicking them feel polished rather than jarring.
document$.subscribe(function () {
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var href = this.getAttribute('href');
      var target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, null, href);
      }
    });
  });
});
