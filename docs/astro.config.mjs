// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import icon from 'astro-icon';

// https://astro.build/config
export default defineConfig({
	site: 'https://docs.interstice.cloud',
	integrations: [
		starlight({
			title: 'Krita Diffusion Plugin',
			logo: { src: './src/assets/logo.png' },
			favicon: '/favicon.png',
			social: {
				github: 'https://github.com/Acly/krita-ai-diffusion',
				discord: 'https://discord.gg/pWyzHfHHhU',
				youtube: 'https://www.youtube.com/@aclysia',
			},
			sidebar: [
				{
					label: 'Setup',
					items: [
						{ label: 'Installation', slug: 'installation' },
						{ label: 'ComfyUI Setup', slug: 'comfyui-setup' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Regions', slug: 'regions' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Base Models', slug: 'base-models' },
						{ label: 'Model Database', slug: 'models' },
					]
				}
			],
			customCss: ['./src/styles/custom.css'],
		}),
		icon()],
});