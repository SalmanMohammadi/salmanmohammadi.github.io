---
layout: index_layout
title: salman mohammadi
---

I'm&nbsp;[Salman Mohammadi](https://www.linkedin.com/in/salman-m-a541a6152/). I love to learn about new things.

<!-- {: class="centered-text"} -->

Artificial intelligence could be the most transformative technology ever created. I'm passionate about ensuring this technology has a positive impact on humanity. See some of my writing below:

<!-- {: class="centered-text"} -->

{%- assign posts = site.content | where_exp: 'post', 'post.hidden != false' -%}
<hr class="contrast footer-hr">
  <div class="post-list">
    <ul class="content-listing">
      {% for post in posts %}    
        <a class="posting-list" href="{{ post.url | prepend: site.baseurl }}"><p class="posting-list">{{ post.title }}</p></a>
      {% endfor %}
        <br>
    </ul>
</div>
