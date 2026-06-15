"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, BrainCircuit, Database, Download, GitBranch, KeyRound, Play, Plus, Save, Upload } from "lucide-react";
import { api, Dataset, LeaderboardRow, MAX_UPLOAD_BYTES, Project, Run } from "@/lib/api";
import { Button, Input, Panel, Select } from "@/components/ui";

const pipelineNodes = ["Data Load", "Normalize", "Block", "Embed", "Similarity", "LLM Review", "Cluster", "Export"];

function metricNumber(metrics: Record<string, unknown>, key: string, fallback = 0) {
  const value = metrics[key];
  return typeof value === "number" ? value : fallback;
}

function formatMetric(value: unknown, digits = 4) {
  return typeof value === "number" ? value.toFixed(digits) : "N/A";
}

function evaluationMetrics(metrics: Record<string, unknown>) {
  const evaluation = metrics.evaluation;
  return evaluation && typeof evaluation === "object" ? evaluation as Record<string, unknown> : undefined;
}

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
  const [projectName, setProjectName] = useState("Customer Entity Resolution");
  const [entityType, setEntityType] = useState("company");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [nameColumn, setNameColumn] = useState("name");
  const [threshold, setThreshold] = useState(0.86);
  const [useLlm, setUseLlm] = useState(false);
  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("gpt-4o-mini");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmMinScore, setLlmMinScore] = useState(0.7);
  const [llmMaxScore, setLlmMaxScore] = useState(0.88);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const selectedProject = projects.find((project) => project.id === selectedProjectId);
  const selectedDataset = datasets.find((dataset) => dataset.id === selectedDatasetId);
  const latestRun = runs[0];
  const metrics = latestRun?.metrics ?? {};
  const evaluation = evaluationMetrics(metrics);

  async function refresh(projectId?: string) {
    const nextProjects = await api.projects();
    setProjects(nextProjects);
    const activeProject = projectId || selectedProjectId || nextProjects[0]?.id || "";
    setSelectedProjectId(activeProject);
    if (activeProject) {
      const [nextDatasets, nextRuns] = await Promise.all([api.datasets(activeProject), api.runs(activeProject)]);
      setDatasets(nextDatasets);
      setRuns(nextRuns);
      setSelectedDatasetId((current) =>
        current && nextDatasets.some((dataset) => dataset.id === current) ? current : nextDatasets[0]?.id || ""
      );
    }
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
    api.leaderboard().then(setLeaderboard).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (selectedProjectId) refresh(selectedProjectId).catch((err) => setError(err.message));
  }, [selectedProjectId]);

  useEffect(() => {
    if (selectedDataset?.columns?.length && !selectedDataset.columns.includes(nameColumn)) {
      setNameColumn(selectedDataset.columns[0]);
    }
  }, [selectedDataset, nameColumn]);

  const previewColumns = useMemo(() => selectedDataset?.columns.slice(0, 6) ?? [], [selectedDataset]);

  async function createProject() {
    setBusy(true);
    setError("");
    try {
      const project = await api.createProject({ name: projectName, entity_type: entityType });
      await refresh(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create project");
    } finally {
      setBusy(false);
    }
  }

  async function upload(file?: File) {
    if (!file || !selectedProjectId) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`File is too large. Uploads are limited to 10 MB; selected file is ${(file.size / 1024 / 1024).toFixed(2)} MB.`);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const dataset = await api.uploadDataset(selectedProjectId, file);
      setSelectedDatasetId(dataset.id);
      await refresh(selectedProjectId);
      setSelectedDatasetId(dataset.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to upload dataset");
    } finally {
      setBusy(false);
    }
  }

  function downloadLatestRun() {
    if (!selectedProjectId || !latestRun) return;
    window.location.href = api.exportRunUrl(selectedProjectId, latestRun.id);
  }

  async function runPipeline() {
    if (!selectedProjectId || !selectedDatasetId) return;
    setBusy(true);
    setError("");
    try {
      await api.startRun(selectedProjectId, {
        dataset_id: selectedDatasetId,
        name_column: nameColumn,
        threshold,
        use_llm: useLlm,
        llm_provider: llmProvider,
        llm_model: llmModel,
        llm_review_min_score: llmMinScore,
        llm_review_max_score: llmMaxScore
      });
      await refresh(selectedProjectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to run pipeline");
    } finally {
      setBusy(false);
    }
  }

  async function saveLlmKey() {
    if (!llmApiKey) return;
    setBusy(true);
    setError("");
    try {
      await api.storeSecret({ provider: llmProvider, api_key: llmApiKey });
      setLlmApiKey("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to store API key");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">OpenMatchER</h1>
            <p className="text-sm text-slate-600">Entity resolution workbench for datasets, models, pipelines, and review.</p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Activity className="h-4 w-4 text-moss" />
            Local-first
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 px-6 py-6 lg:grid-cols-[290px_1fr]">
        <aside className="space-y-4">
          <Panel>
            <div className="mb-3 flex items-center gap-2 font-medium"><Plus className="h-4 w-4" />Project</div>
            <div className="space-y-3">
              <Input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
              <Select value={entityType} onChange={(event) => setEntityType(event.target.value)}>
                <option value="person">Person names</option>
                <option value="company">Company names</option>
                <option value="organization">Organizations</option>
                <option value="product">Products</option>
                <option value="custom">Custom entities</option>
              </Select>
              <Button className="w-full" disabled={busy} onClick={createProject}><Plus className="h-4 w-4" />Create</Button>
            </div>
          </Panel>

          <Panel>
            <div className="mb-3 flex items-center gap-2 font-medium"><Database className="h-4 w-4" />Workspace</div>
            <Select value={selectedProjectId} onChange={(event) => setSelectedProjectId(event.target.value)}>
              <option value="">Select project</option>
              {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
            </Select>
            <label className="mt-3 flex h-24 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-slate-300 bg-mist text-sm text-slate-600">
              <Upload className="mb-2 h-5 w-5" />
              Upload CSV, JSON, or Parquet
              <input className="hidden" type="file" accept=".csv,.json,.parquet" onChange={(event) => upload(event.target.files?.[0])} />
            </label>
            <div className="mt-2 text-xs text-slate-500">Maximum file size: 10 MB</div>
          </Panel>

          <Panel>
            <div className="mb-3 flex items-center gap-2 font-medium"><KeyRound className="h-4 w-4" />LLM Providers</div>
            <div className="space-y-3">
              <Select value={llmProvider} onChange={(event) => setLlmProvider(event.target.value)}>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="gemini">Gemini</option>
                <option value="ollama">Ollama</option>
                <option value="openrouter">OpenRouter</option>
              </Select>
              <Input value={llmModel} onChange={(event) => setLlmModel(event.target.value)} placeholder="Model name" />
              <Input
                type="password"
                value={llmApiKey}
                onChange={(event) => setLlmApiKey(event.target.value)}
                placeholder="Provider API key"
              />
              <Button className="w-full" disabled={busy || !llmApiKey} onClick={saveLlmKey}><Save className="h-4 w-4" />Store key</Button>
            </div>
          </Panel>
        </aside>

        <section className="space-y-5">
          {error && <div className="rounded-md border border-coral bg-white px-4 py-3 text-sm text-coral">{error}</div>}
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Projects", projects.length],
              ["Datasets", datasets.length],
              ["Runs", runs.length],
              ["Clusters", metricNumber(metrics, "cluster_count")]
            ].map(([label, value]) => (
              <Panel key={label as string}>
                <div className="text-sm text-slate-500">{label}</div>
                <div className="mt-2 text-3xl font-semibold">{value}</div>
              </Panel>
            ))}
          </div>

          <Panel>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Resolution Pipeline</h2>
                <p className="text-sm text-slate-600">{selectedProject?.name ?? "Create a project to begin"}</p>
              </div>
              <div className="flex gap-2">
                <Button disabled={busy || !selectedDatasetId} onClick={runPipeline}><Play className="h-4 w-4" />Run</Button>
                <Button className="bg-steel hover:bg-moss" disabled={!latestRun} onClick={downloadLatestRun}>
                  <Download className="h-4 w-4" />CSV
                </Button>
              </div>
            </div>
            <div className="mb-4 grid gap-3 md:grid-cols-3">
              <Select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)}>
                <option value="">Select dataset</option>
                {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.filename}</option>)}
              </Select>
              <Select value={nameColumn} onChange={(event) => setNameColumn(event.target.value)}>
                {(selectedDataset?.columns ?? ["name"]).map((column) => <option key={column} value={column}>{column}</option>)}
              </Select>
              <Input type="number" min="0" max="1" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            </div>
            <div className="mb-4 grid gap-3 rounded-md border border-slate-200 bg-mist p-3 md:grid-cols-[1fr_120px_120px]">
              <label className="flex items-center gap-2 text-sm font-medium">
                <input className="h-4 w-4 accent-moss" type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
                <BrainCircuit className="h-4 w-4 text-moss" />
                LLM review for uncertain pairs
              </label>
              <Input type="number" min="0" max="1" step="0.01" value={llmMinScore} onChange={(event) => setLlmMinScore(Number(event.target.value))} />
              <Input type="number" min="0" max="1" step="0.01" value={llmMaxScore} onChange={(event) => setLlmMaxScore(Number(event.target.value))} />
            </div>
            <div className="grid gap-2 md:grid-cols-8">
              {pipelineNodes.map((node) => (
                <div key={node} className="min-h-20 rounded-md border border-slate-200 bg-mist p-3 text-sm">
                  <GitBranch className="mb-2 h-4 w-4 text-steel" />
                  {node}
                </div>
              ))}
            </div>
          </Panel>

          <div className="grid gap-5 xl:grid-cols-2">
            <Panel className="overflow-x-auto">
              <h2 className="mb-3 text-lg font-semibold">Dataset Preview</h2>
              <table className="w-full min-w-[520px] text-left text-sm">
                <thead><tr>{previewColumns.map((column) => <th className="border-b p-2" key={column}>{column}</th>)}</tr></thead>
                <tbody>
                  {(selectedDataset?.preview ?? []).slice(0, 8).map((row, index) => (
                    <tr key={index}>{previewColumns.map((column) => <td className="border-b p-2 text-slate-700" key={column}>{String(row[column] ?? "")}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </Panel>

            <Panel>
              <h2 className="mb-3 text-lg font-semibold">Cluster Review</h2>
              <div className="space-y-3">
                {(latestRun?.results.matches ?? []).slice(0, 6).map((match) => (
                  <div key={`${match.left_index}-${match.right_index}`} className="rounded-md border border-slate-200 p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span>Rows {match.left_index + 1} and {match.right_index + 1}</span>
                      <span className="font-semibold text-moss">{Math.round(match.final_score * 100)}%</span>
                    </div>
                    {"llm_review" in match && (
                      <div className="mt-2 text-xs font-medium text-steel">
                        {(match as { llm_review?: { reviewed?: boolean } }).llm_review?.reviewed ? "LLM reviewed" : "Deterministic"}
                      </div>
                    )}
                    <p className="mt-2 text-xs text-slate-600">{match.reasoning}</p>
                  </div>
                ))}
                {!latestRun && <p className="text-sm text-slate-600">Run a pipeline to review matched clusters.</p>}
              </div>
            </Panel>
          </div>

          <Panel className="overflow-x-auto">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold">Latest Dataset Result</h2>
                <p className="text-sm text-slate-600">
                  {evaluation
                    ? `Evaluated against ${evaluation.label_column} labels from ${selectedDataset?.filename ?? "the uploaded dataset"}.`
                    : "Run metrics from the currently selected uploaded dataset."}
                </p>
              </div>
              <div className="text-sm text-slate-600">{latestRun ? latestRun.status : "No run yet"}</div>
            </div>
            {evaluation ? (
              <table className="mb-6 w-full min-w-[520px] text-left text-sm">
                <thead>
                  <tr>
                    <th className="border-b p-2">Dataset</th>
                    <th className="border-b p-2">Precision</th>
                    <th className="border-b p-2">Recall</th>
                    <th className="border-b p-2">F1</th>
                    <th className="border-b p-2">TP</th>
                    <th className="border-b p-2">FP</th>
                    <th className="border-b p-2">FN</th>
                    <th className="border-b p-2">Gold Pairs</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border-b p-2 font-medium">{selectedDataset?.filename ?? "Selected dataset"}</td>
                    <td className="border-b p-2">{formatMetric(evaluation.precision)}</td>
                    <td className="border-b p-2">{formatMetric(evaluation.recall)}</td>
                    <td className="border-b p-2">{formatMetric(evaluation.f1)}</td>
                    <td className="border-b p-2">{String(evaluation.true_positive ?? "N/A")}</td>
                    <td className="border-b p-2">{String(evaluation.false_positive ?? "N/A")}</td>
                    <td className="border-b p-2">{String(evaluation.false_negative ?? "N/A")}</td>
                    <td className="border-b p-2">{String(evaluation.gold_pairs ?? "N/A")}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <table className="mb-6 w-full min-w-[520px] text-left text-sm">
                <thead>
                  <tr>
                    <th className="border-b p-2">Dataset</th>
                    <th className="border-b p-2">Records</th>
                    <th className="border-b p-2">Candidate Pairs</th>
                    <th className="border-b p-2">Matches</th>
                    <th className="border-b p-2">Clusters</th>
                    <th className="border-b p-2">Average Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border-b p-2 font-medium">{selectedDataset?.filename ?? "Selected dataset"}</td>
                    <td className="border-b p-2">{metricNumber(metrics, "record_count")}</td>
                    <td className="border-b p-2">{metricNumber(metrics, "candidate_pairs")}</td>
                    <td className="border-b p-2">{metricNumber(metrics, "match_count")}</td>
                    <td className="border-b p-2">{metricNumber(metrics, "cluster_count")}</td>
                    <td className="border-b p-2">{formatMetric(metrics.average_score)}</td>
                  </tr>
                </tbody>
              </table>
            )}

            <h2 className="mb-3 text-lg font-semibold">Demo Benchmark Baseline</h2>
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead>
                <tr>
                  <th className="border-b p-2">Model</th>
                  <th className="border-b p-2">Micro Precision</th>
                  <th className="border-b p-2">Micro Recall</th>
                  <th className="border-b p-2">Micro F1</th>
                  <th className="border-b p-2">Macro F1</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => (
                  <tr key={row.model}>
                    <td className="border-b p-2 font-medium">{row.model}</td>
                    <td className="border-b p-2">{(row.micro_precision ?? row.precision).toFixed(4)}</td>
                    <td className="border-b p-2">{(row.micro_recall ?? row.recall).toFixed(4)}</td>
                    <td className="border-b p-2">{(row.micro_f1 ?? row.f1).toFixed(4)}</td>
                    <td className="border-b p-2">{(row.macro_f1 ?? row.f1).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </section>
      </div>
    </main>
  );
}
