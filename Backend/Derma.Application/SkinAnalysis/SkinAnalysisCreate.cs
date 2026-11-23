
using Derma.Infrastructure;
using FluentValidation;
using MediatR;
using System;
using System.Diagnostics;
using System.IO;
using System.Security.Claims;
using System.Threading;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisCreate
    {
        public class CommandValidator : AbstractValidator<Command>
        {
            public CommandValidator()
            {
                RuleFor(x => x.SkinAnalysis).SetValidator(new SkinAnalysisValidator());
            }
        }
        public class Command : IRequest<Unit>
        {
            public required Derma.Domain.SkinAnalysis SkinAnalysis { get; set; }
        }

        public class Handler : IRequestHandler<Command, Unit>
        {
            private readonly DataContext _context;
            private readonly Microsoft.AspNetCore.Http.IHttpContextAccessor _httpContextAccessor;

            public Handler(DataContext context, Microsoft.AspNetCore.Http.IHttpContextAccessor httpContextAccessor)
            {
                _context = context;
                _httpContextAccessor = httpContextAccessor;
            }

            public async Task<Unit> Handle(
                Command request,
                CancellationToken cancellationToken)
            {


                var userId = _httpContextAccessor.HttpContext?
                    .User.FindFirst(ClaimTypes.NameIdentifier)?.Value;

                if (!Guid.TryParse(userId, out var id))
                    throw new UnauthorizedAccessException("Nieprawidłowy UserId.");

                var skinAnalysis = request.SkinAnalysis;

                skinAnalysis.userId = id;

                await _context.SkinAnalysis.AddAsync(
                    skinAnalysis,
                    cancellationToken
                );

                await _context.SaveChangesAsync(cancellationToken);

                var projectRoot = FindProjectRoot();
                var pythonPathFile = Path.Combine(projectRoot, "Models", "prod", ".derma-python-exe");

                var pythonExe = Environment.GetEnvironmentVariable("DERMA_PYTHON_EXE");
                if (string.IsNullOrWhiteSpace(pythonExe) && File.Exists(pythonPathFile))
                    pythonExe = File.ReadAllText(pythonPathFile).Trim();

                if (string.IsNullOrWhiteSpace(pythonExe) || !File.Exists(pythonExe))
                    throw new FileNotFoundException(
                        "Conda Python interpreter not found. Run Models/prod/build_env.ps1 first."
                    );

                var pythonScript = Environment.GetEnvironmentVariable("DERMA_PYTHON_SCRIPT");
                if (string.IsNullOrWhiteSpace(pythonScript))
                    pythonScript = Path.Combine(
                        projectRoot, "Models", "prod", "process_pipeline_api.py"
                    );

                pythonScript = Path.GetFullPath(pythonScript);
                if (!File.Exists(pythonScript))
                    throw new FileNotFoundException(
                        $"Python pipeline script not found: {pythonScript}"
                    );

                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    WorkingDirectory = Path.GetDirectoryName(pythonScript)!,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                startInfo.ArgumentList.Add(pythonScript);
                startInfo.ArgumentList.Add("--analysis-id");
                startInfo.ArgumentList.Add(skinAnalysis.Id.ToString());
                startInfo.Environment["USER_ID"] = id.ToString();

                Process.Start(startInfo);

                return Unit.Value;
            }
            
            private static string FindProjectRoot()
            {
                var directory = new DirectoryInfo(AppContext.BaseDirectory);

                while (directory != null)
                {
                    if (File.Exists(Path.Combine(directory.FullName, "Models", "prod", "environment.yml")))
                        return directory.FullName;

                    directory = directory.Parent;
                }

                throw new DirectoryNotFoundException(
                    "Could not locate the DermaAI project root directory."
                );
            }
        }
    }
}
