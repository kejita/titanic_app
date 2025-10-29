document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('sample-form');
  if (!form) return;
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form));
    console.log('Form submitted:', data);
    alert('フォームを送信しました。コンソールを確認してください。');
  });
});

