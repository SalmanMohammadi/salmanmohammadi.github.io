---
layout: index_layout
title: salman mohammadi
---

I'm&nbsp;[Salman Mohammadi](https://www.linkedin.com/in/salman-m-a541a6152/). I love to learn about new things.

<!-- {: class="centered-text"} -->

<!-- {: class="centered-text"} -->

Artificial intelligence could be the most transformative technology ever created. I'd like to make sure it has a positive impact on humanity. See some of my writing below:

<!-- {: class="centered-text"} -->

{%- assign posts = site.content | where_exp: 'post', 'post.hidden != false' -%}

<hr class="contrast footer-hr">
  <div class="post-list">
    <ul class="content-listing">
      {% for post in posts %}
        {% if post.external_url %}
            <a class="posting-list" href="{{ post.external_url }}" target="_blank">
              <p class="posting-list">{{ post.title }}</p>
            </a>
          {% else %}
            <a class="posting-list" href="{{ post.url | prepend: site.baseurl }}"><p class="posting-list">{{ post.title }}</p></a>
        {% endif %}
      {% endfor %}
      <a class="posting-list" href="https://huggingface.co/blog/accelerate-nd-parallel" target="_blank">
        <p class="posting-list">Accelerate ND-Parallel: A guide to Efficient Multi-GPU Training</p>
      </a>
      <a class="posting-list" href="https://huggingface.co/blog/liger-grpo" target="_blank">
        <p class="posting-list">Liger GRPO meets TRL</p>
      </a>
      <a class="posting-list" href="https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm" target="_blank">
        <p class="posting-list">Training Large Language Models with Interpreter Feedback using WebAssembly</p>
      </a>
      <a class="posting-list" href="https://axolotlai.substack.com/p/process-reward-models" target="_blank">
        <p class="posting-list">Process Reward Models</p>
      </a>
        <br>
    </ul>
</div>
