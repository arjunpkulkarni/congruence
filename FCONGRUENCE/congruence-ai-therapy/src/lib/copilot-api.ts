// Congruence Copilot API Client

const BASE_URL = 'https://api.congruenceinsights.com';

export interface AgentStatus {
  status: 'ready' | 'error' | 'loading';
  model: string;
  tools_count: number;
  message: string;
}

export interface ChatRequest {
  message: string;
  user_id: string;
  role: 'clinician' | 'admin' | 'practice_owner';
  context: {
    selected_patient?: string;
    selected_session?: string;
    [key: string]: any;
  };
}

export interface AgentAction {
  type: string;
  label: string;
  data: any;
}

export interface ChatResponse {
  response: string;
  actions?: AgentAction[];
  tools_used: string[];
  context: {
    selected_patient?: string;
    selected_session?: string;
    [key: string]: any;
  };
  metadata: {
    model_used: string;
    [key: string]: any;
  };
}

export class CongruenceAgentAPI {
  private baseUrl: string;

  constructor(baseUrl: string = BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getStatus(): Promise<AgentStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/agent/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to get agent status:', error);
      throw error;
    }
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/agent/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }
}

export const agentAPI = new CongruenceAgentAPI();
