---
layout: post
title: salman mohammadi
---

I'm&nbsp;[Salman](https://www.linkedin.com/in/salman-m-a541a6152/). I love to learn about new things.
{: class="centered-text"}

Artificial intelligence could be the most transformative technology ever created. I'm passionate about ensuring this technology has a positive impact on humanity.
{: class="centered-text"}

  <div class="post-list">
    <ul class="content-listing">
      {% for post in site.content %}      
        <br>
        <hr class="contrast centered-text">
        <a href="{{ post.url | prepend: site.baseurl }}"><p class="centered-text ">{{ post.title }}</p></a>
      {% endfor %}
        <br>
    </ul>
</div>