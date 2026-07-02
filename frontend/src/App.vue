<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">近</div>
        <div>
          <strong>fnirs与深度学习</strong>
          <span>多智能体自助平台</span>
        </div>
      </div>
      <nav class="nav-list">
        <RouterLink v-for="item in navItems" :key="item.to" :to="item.to">
          <span class="nav-icon">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
      <div class="sidebar-footer">
        <span :class="['status-dot', healthClass]"></span>
        <span>本地模型：{{ statusText(store.health?.ollama_status || "checking") }}</span>
      </div>
    </aside>

    <main class="app-main">
      <RouterView />
    </main>
    <FeedbackHost />
  </div>
</template>

<script setup>
import { computed, onMounted } from "vue";
import { RouterLink, RouterView } from "vue-router";
import FeedbackHost from "./components/FeedbackHost.vue";
import { usePlatformStore } from "./stores/platform";
import { statusText } from "./formatters";

const store = usePlatformStore();

const navItems = [
  { to: "/", label: "总览", icon: "览" },
  { to: "/chat", label: "对话", icon: "问" },
  { to: "/knowledge", label: "知识库", icon: "知" },
  { to: "/data", label: "数据", icon: "数" },
  { to: "/experiments", label: "实验", icon: "验" },
  { to: "/results", label: "结果", icon: "果" },
  { to: "/settings", label: "设置", icon: "设" },
];

const healthClass = computed(() => (store.health?.ollama_status === "healthy" ? "ok" : "warn"));

onMounted(() => {
  store.refreshAll();
});
</script>
