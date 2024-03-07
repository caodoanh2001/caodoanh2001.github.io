---
layout: page
permalink: /publications/
title: publications
description: Feel free to visit my Google Scholar profile at https://scholar.google.com/citations?user=WHviN4AAAAAJ&hl=vi&oi=ao
years: [2021, 2022, 2023, 2024]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>