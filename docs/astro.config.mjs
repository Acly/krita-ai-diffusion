// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import icon from 'astro-icon';

// https://astro.build/config
export default defineConfig({
	site: 'https://docs.interstice.cloud',
	integrations: [
		starlight({
			title: 'Krita AI Handbook',
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
						{ label: 'Common Issues', slug: 'common-issues' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'First Steps', slug: 'basics' },
						{ label: 'Selection Fill', slug: 'selections' },
						{ label: 'Text Prompts', slug: 'prompts' },
						{ label: 'Control Layers', slug: 'control-layers' },
						{ label: 'Regions', slug: 'regions' },
						{ label: 'Samplers', slug: 'samplers' },
						{ label: 'Custom Graphs', slug: 'custom-graph' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Resolutions', slug: 'resolutions' },
						{ label: 'Base Models', slug: 'base-models' },
						{ label: 'Model Database', slug: 'models' },
					]
				}
			],
			customCss: ['./src/styles/custom.css'],
			components: {
				Header: './src/components/Header.astro',
				Hero: './src/components/Hero.astro',
			}
		}),
		icon()],
});