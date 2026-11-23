using Derma.Application.AnalysisResult;
using Derma.Application.AnalysisResult;
using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;

namespace Derma.API.Controllers
{
    
    public class AnalysisResultController : BaseApiController
    {
        private readonly IMediator _mediator;
        public AnalysisResultController(IMediator mediator)
        {
            _mediator = mediator;
        }
        private Guid CurrentUserId =>
    Guid.TryParse(User.FindFirstValue(ClaimTypes.NameIdentifier), out var id) ? id : Guid.Empty;
        private bool IsAdmin => User.IsInRole("Admin");
        [HttpGet]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<List<AnalysisResult>>> GetAnalysisResult()
        {
            return await Mediator.Send(new AnalysisResultList.Query
            {
                UserId = CurrentUserId,
                IsAdmin = IsAdmin
            });
        }
        [HttpGet("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<AnalysisResult>> GetAnalysisResult(Guid id)
        {
            return await Mediator.Send(new AnalysisResultDetails.Query {Id = id, UserId = CurrentUserId, IsAdmin = IsAdmin });
        }
        [HttpPost]
        [Authorize(AuthenticationSchemes = "ApiKey")]
        public async Task<IActionResult> CreateAnalysisResult(AnalysisResult analysisResult)
        {
            await Mediator.Send(
                new AnalysisResultCreate.Command { AnalysisResult = analysisResult });

            return Ok();
        }

        [HttpDelete("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<IActionResult> DeleteAnalysisResult(Guid id)
        {
            await Mediator.Send(
                new AnalysisResultDelete.Command { Id = id, UserId = CurrentUserId, IsAdmin = IsAdmin });

            return Ok();
        }
        
    }
}
