declare module 'default_api' {
  export function read_file(params: { path: string }): Promise<any>;
  export function natural_language_write_file(params: {
    path: string;
    prompt: string;
    language?: string | null;
    selected_content?: string | null;
  }): Promise<any>;
  export function delete_file(params: { path: string }): Promise<any>;
  export function run_terminal_command(params: { command: string }): Promise<any>;
  export function list_project_files(params: { path: string }): Promise<any>;
}