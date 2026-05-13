import { RooCodeSettings } from "@roo-code/types"

import type { SupportedProvider } from "@/types/index.js"

const envVarMap: Record<SupportedProvider, string> = {
	anthropic: "ANTHROPIC_API_KEY",
	openai: "OPENAI_API_KEY",
	"openai-native": "OPENAI_API_KEY",
	gemini: "GOOGLE_API_KEY",
	openrouter: "OPENROUTER_API_KEY",
	"vercel-ai-gateway": "VERCEL_AI_GATEWAY_API_KEY",
	roo: "ROO_API_KEY",
}

export function getEnvVarName(provider: SupportedProvider): string {
	return envVarMap[provider]
}

export function getApiKeyFromEnv(provider: SupportedProvider): string | undefined {
	const envVar = getEnvVarName(provider)
	return process.env[envVar]
}

export function getProviderSettings(
	provider: SupportedProvider,
	apiKey: string | undefined,
	model: string | undefined,
	baseUrl?: string,
	contextWindow?: number,
	temperature?: number,
	topP?: number,
	topK?: number,
): RooCodeSettings {
	const config: RooCodeSettings = { apiProvider: provider }
	if (temperature !== undefined) config.modelTemperature = temperature
	if (topP !== undefined) config.modelTopP = topP
	if (topK !== undefined) config.modelTopK = topK

	switch (provider) {
		case "anthropic":
			if (apiKey) config.apiKey = apiKey
			if (model) config.apiModelId = model
			break
		case "openai":
			if (apiKey) config.openAiApiKey = apiKey
			if (model) config.openAiModelId = model
			if (baseUrl) config.openAiBaseUrl = baseUrl
			if (contextWindow) {
				config.openAiCustomModelInfo = {
					maxTokens: 8192,
					contextWindow,
					supportsImages: false,
					supportsPromptCache: false,
					inputPrice: 0,
					outputPrice: 0,
				}
			}
			break
		case "openai-native":
			if (apiKey) config.openAiNativeApiKey = apiKey
			if (model) config.apiModelId = model
			break
		case "gemini":
			if (apiKey) config.geminiApiKey = apiKey
			if (model) config.apiModelId = model
			break
		case "openrouter":
			if (apiKey) config.openRouterApiKey = apiKey
			if (model) config.openRouterModelId = model
			break
		case "vercel-ai-gateway":
			if (apiKey) config.vercelAiGatewayApiKey = apiKey
			if (model) config.vercelAiGatewayModelId = model
			break
		case "roo":
			if (apiKey) config.rooApiKey = apiKey
			if (model) config.apiModelId = model
			break
		default:
			if (apiKey) config.apiKey = apiKey
			if (model) config.apiModelId = model
	}

	return config
}
